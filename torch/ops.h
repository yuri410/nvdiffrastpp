#pragma once

#include "torch_bindings.h"
#include "torch_bindings_gl.h"

inline int GetCurrentCUDADeviceIndex(torch::Device device)
{
	int cuda_device_idx;
	if (device == nullptr)
	{
		cuda_device_idx = at::cuda::current_device();
	}
	else
	{
		torch::DeviceGuard scope(device);
		cuda_device_idx = at::cuda::current_device();
	}
	return cuda_device_idx;
}

class DepthPeeler;
class RasterizeBaseContext
{
public:
	virtual bool isGLContext() const = 0;

protected:
	RasterizeBaseContext(bool output_db)
		: m_output_db(output_db)
	{ }

public:
	bool m_output_db;
	DepthPeeler* m_active_depth_peeler = nullptr;
};

class RasterizeCudaContext : public RasterizeBaseContext
{
public:
	/*
	 * Create a new Cuda rasterizer context.
	 *
	 *  The context is deleted and internal storage is released when the object is
	 *  destroyed.
	 *
	 *  Args:
	 *    device (Optional): Cuda device on which the context is created. Type can be
	 *                       `torch.device`, string (e.g., `'cuda:1'`), or int. If not
	 *                       specified, context will be created on currently active Cuda
	 *                       device.
	 *  Returns:
	 *    The newly created Cuda rasterizer context.
	 */
	RasterizeCudaContext(torch::Device device)
		: RasterizeBaseContext(true)
		, m_cpp_wrapper(GetCurrentCUDADeviceIndex(device))
	{
		
	}

	bool isGLContext() const override { return false; }

//private:
	RasterizeCRStateWrapper m_cpp_wrapper;
};

class RasterizeGLContext : public RasterizeBaseContext
{
public:
	enum Mode
	{
		Mode_Automatic,
		Mode_Manual
	};

	/*
	 * Create a new OpenGL rasterizer context.
	 *
	 *	Creating an OpenGL context is a slow operation so you should usually reuse the same
	 *	context in all calls to `rasterize()` on the same CPU thread. The OpenGL context
	 *	is deleted when the object is destroyed.
	 *
	 *	Side note: When using the OpenGL context in a rasterization operation, the
	 *	context's internal framebuffer object is automatically enlarged to accommodate the
	 *	rasterization operation's output shape, but it is never shrunk in size until the
	 *	context is destroyed. Thus, if you need to rasterize, say, deep low-resolution
	 *	tensors and also shallow high-resolution tensors, you can conserve GPU memory by
	 *	creating two separate OpenGL contexts for these tasks. In this scenario, using the
	 *	same OpenGL context for both tasks would end up reserving GPU memory for a deep,
	 *	high-resolution output tensor.
	 *
	 *	Args:
	 *	  output_db (bool): Compute and output image-space derivates of barycentrics.
	 *	  mode: OpenGL context handling mode. Valid values are 'manual' and 'automatic'.
	 *	  device (Optional): Cuda device on which the context is created. Type can be
	 *						 `torch.device`, string (e.g., `'cuda:1'`), or int. If not
	 *						 specified, context will be created on currently active Cuda
	 *						 device.
	 *	Returns:
	 *	  The newly created OpenGL rasterizer context.
	 */
	RasterizeGLContext(bool output_db, Mode mode, torch::Device device)
		: RasterizeBaseContext(output_db)
		, m_cpp_wrapper(output_db, mode == Mode_Automatic, GetCurrentCUDADeviceIndex(device))
		, m_mode(mode)
	{
		
	}

	void set_context()
	{
		// Set (activate) OpenGL context in the current CPU thread.
        // Only available if context was created in manual mode.
		assert(m_mode == Mode_Manual);
		m_cpp_wrapper.setContext();
	}

	void release_context()
	{
		// Release(deactivate) currently active OpenGL context.
		// Only available if context was created in manual mode.
		assert(m_mode == Mode_Manual);
		m_cpp_wrapper.releaseContext();
	}

	bool isGLContext() const override { return true; }

//private:

	Mode m_mode;
	RasterizeGLStateWrapper m_cpp_wrapper;
};

class _rasterize_func : public torch::autograd::Function<_rasterize_func>
{
public:
	
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, RasterizeBaseContext* raster_ctx, torch::Tensor pos, torch::Tensor tri, std::tuple<int,int> resolution, torch::Tensor ranges, bool grad_db, int peeling_idx)
	{
		torch::Tensor out;
		torch::Tensor out_db;

		if (raster_ctx->isGLContext())
		{
			RasterizeGLContext* raster_gl_ctx = static_cast<RasterizeGLContext*>(raster_ctx);
			auto result = rasterize_fwd_gl(raster_gl_ctx->m_cpp_wrapper, pos, tri, resolution, ranges, peeling_idx);
			out = std::get<0>(result);
			out_db = std::get<1>(result);
		}
		else
		{
			RasterizeCudaContext* raster_cuda_ctx = static_cast<RasterizeCudaContext*>(raster_ctx);
			auto result = rasterize_fwd_cuda(raster_cuda_ctx->m_cpp_wrapper, pos, tri, resolution, ranges, peeling_idx);
			out = std::get<0>(result);
			out_db = std::get<1>(result);
		}

		ctx->save_for_backward({ pos, tri, out });
		
		ctx->saved_data["grad_db"] = grad_db;

		return { out, out_db };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];
		torch::Tensor ddb = grad_output[1];

		torch::Tensor pos, tri, out;

		pos = ctx->get_saved_variables()[0];
		tri = ctx->get_saved_variables()[1];
		out = ctx->get_saved_variables()[2];
		
		torch::Tensor g_pos;
		bool grad_db = ctx->saved_data["grad_db"].toBool();
		if (grad_db)
		{
			g_pos = rasterize_grad_db(pos, tri, out, dy, ddb);
		}
		else
		{
			g_pos = rasterize_grad(pos, tri, out, dy);
		}

		return { torch::Tensor(), g_pos, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor() };
	}
};

/*
 * Rasterize triangles.
 *
 * All input tensors must be contiguous and reside in GPU memory except for
 * the `ranges` tensor that, if specified, has to reside in CPU memory. The
 * output tensors will be contiguous and reside in GPU memory.
 *
 * Args:
 * 	glctx: Rasterizer context of type `RasterizeGLContext` or `RasterizeCudaContext`.
 * 	pos: Vertex position tensor with dtype `torch.float32`. To enable range
 * 		 mode, this tensor should have a 2D shape [num_vertices, 4]. To enable
 * 		 instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
 * 	tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
 * 	resolution: Output resolution as integer tuple (height, width).
 * 	ranges: In range mode, tensor with shape [minibatch_size, 2] and dtype
 * 			`torch.int32`, specifying start indices and counts into `tri`.
 * 			Ignored in instanced mode.
 * 	grad_db: Propagate gradients of image-space derivatives of barycentrics
 * 			 into `pos` in backward pass. Ignored if using an OpenGL context that
 * 			 was not configured to output image-space derivatives.
 *
 * Returns:
 * 	A tuple of two tensors. The first output tensor has shape [minibatch_size,
 * 	height, width, 4] and contains the main rasterizer output in order (u, v, z/w,
 * 	triangle_id). If the OpenGL context was configured to output image-space
 * 	derivatives of barycentrics, the second output tensor will also have shape
 * 	[minibatch_size, height, width, 4] and contain said derivatives in order
 * 	(du/dX, du/dY, dv/dX, dv/dY). Otherwise it will be an empty tensor with shape
 * 	[minibatch_size, height, width, 0].
 */
inline torch::autograd::variable_list rasterize(RasterizeBaseContext* glctx, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges = torch::Tensor(), bool grad_db = true)
{
	grad_db &= glctx->m_output_db;

	// Sanitize inputs.
	if (ranges.size(0) == 0)
	{
		ranges = torch::empty({ 0,2 }, c10::TensorOptions(c10::ScalarType::Int)).cpu();
	}

	if (glctx->m_active_depth_peeler != nullptr)
		throw std::exception("Cannot call rasterize() during depth peeling operation, use rasterize_next_layer() instead");

	// Instantiate the function.
	return _rasterize_func::apply(glctx, pos, tri, resolution, ranges, grad_db, -1);
}

// ----------------------------------------------------------------------------
//  Depth peeler context manager for rasterizing multiple depth layers.
// ----------------------------------------------------------------------------
class DepthPeeler
{
public:
	DepthPeeler(RasterizeBaseContext* glctx, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges = torch::Tensor(), bool grad_db = true)
	{
		/*
		 * Create a depth peeler object for rasterizing multiple depth layers.
		 * 
		 * Arguments are the same as in `rasterize()`.
		 * 
		 * Returns:
		 *   The newly created depth peeler.
		 */
		grad_db &= glctx->m_output_db;

		// Sanitize inputs as usual.
		if (ranges.size(0) == 0)
		{
			ranges = torch::empty({ 0,2 }, c10::TensorOptions(c10::ScalarType::Int)).cpu();
		}

		// Store all the parameters.
		m_raster_ctx = glctx;
		m_pos = pos;
		m_tri = tri;
		m_resolution = resolution;
		m_ranges = ranges;
		m_grad_db = grad_db;
		m_peeling_idx = -1;
	}

	void _enter()
	{
		if (m_raster_ctx == nullptr)
			throw std::exception("Cannot re-enter a terminated depth peeling operation");
		if (m_raster_ctx->m_active_depth_peeler != nullptr)
			throw std::exception("Cannot have multiple depth peelers active simultaneously in a rasterization context");

		m_raster_ctx->m_active_depth_peeler = this;
		m_peeling_idx = 0;
	}

	void _exit()
	{
		assert(m_raster_ctx->m_active_depth_peeler == this);
		m_raster_ctx->m_active_depth_peeler = nullptr;
		m_raster_ctx = nullptr;
	}

	torch::autograd::variable_list rasterize_next_layer()
	{
		/*
		 * Rasterize next depth layer.
		 * 
		 * Operation is equivalent to `rasterize()` except that previously reported
		 * surface points are culled away.
		 * 
		 * Returns:
		 *   A tuple of two tensors as in `rasterize()`.
		 */
		assert(m_peeling_idx >= 0);
		torch::autograd::variable_list result = _rasterize_func::apply(m_raster_ctx, m_pos, m_tri, m_resolution, m_ranges, m_grad_db, m_peeling_idx);
		m_peeling_idx++;
		return result;
	}

	RasterizeBaseContext* m_raster_ctx;
	torch::Tensor m_pos;
	torch::Tensor m_tri;
	std::tuple<int, int> m_resolution;
	torch::Tensor m_ranges;
	bool m_grad_db;
	int m_peeling_idx;
};

//----------------------------------------------------------------------------
// Interpolate.
//----------------------------------------------------------------------------

// Output pixel differentials for at least some attributes.
class _interpolate_func_da : public torch::autograd::Function<_interpolate_func_da>
{
public:

	static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db, bool diff_attrs_all, std::vector<int>& diff_attrs_list)
	{
		auto result = interpolate_fwd_da(attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list);

		torch::Tensor out = std::get<0>(result);
		torch::Tensor out_da = std::get<1>(result);

		ctx->save_for_backward({ attr, rast, tri, rast_db });
		ctx->saved_data["diff_attrs_all"] = diff_attrs_all;
		ctx->saved_data["diff_attrs_list"] = diff_attrs_list;

		return { out, out_da };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];
		torch::Tensor dda = grad_output[1];

		torch::Tensor attr = ctx->get_saved_variables()[0];
		torch::Tensor rast = ctx->get_saved_variables()[1];
		torch::Tensor tri = ctx->get_saved_variables()[2];
		torch::Tensor rast_db = ctx->get_saved_variables()[3];

		bool diff_attrs_all = ctx->saved_data["diff_attrs_all"].toBool();
		std::vector<int> diff_attrs_list;
		{
			auto temp = ctx->saved_data["diff_attrs_list"].toIntVector();
			for (auto e : temp)
				diff_attrs_list.push_back((int)e);
		}

		auto result = interpolate_grad_da(attr, rast, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list);

		torch::Tensor g_attr = std::get<0>(result);
		torch::Tensor g_rast = std::get<1>(result);
		torch::Tensor g_rast_db = std::get<2>(result);

		return { g_attr, g_rast, torch::Tensor(), g_rast_db, torch::Tensor(), torch::Tensor() };
	}
};

// No pixel differential for any attribute.
class _interpolate_func : public torch::autograd::Function<_interpolate_func>
{
public:
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, torch::Tensor attr, torch::Tensor rast, torch::Tensor tri)
	{
		auto result = interpolate_fwd(attr, rast, tri);

		torch::Tensor out = std::get<0>(result);
		torch::Tensor out_da = std::get<1>(result);

		ctx->save_for_backward({ attr, rast, tri });

		return { out, out_da };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];
		torch::Tensor dda = grad_output[1];

		torch::Tensor attr = ctx->get_saved_variables()[0];
		torch::Tensor rast = ctx->get_saved_variables()[1];
		torch::Tensor tri = ctx->get_saved_variables()[2];

		auto result = interpolate_grad(attr, rast, tri, dy);

		torch::Tensor g_attr = std::get<0>(result);
		torch::Tensor g_rast = std::get<1>(result);

		return { g_attr, g_rast, torch::Tensor() };
	}
};

/*
 * Interpolate vertex attributes.
 *
 *	All input tensors must be contiguous and reside in GPU memory. The output tensors
 *	will be contiguous and reside in GPU memory.
 *
 *	Args:
 *		attr: Attribute tensor with dtype `torch.float32`.
 *			  Shape is [num_vertices, num_attributes] in range mode, or
 *			  [minibatch_size, num_vertices, num_attributes] in instanced mode.
 *			  Broadcasting is supported along the minibatch axis.
 *		rast: Main output tensor from `rasterize()`.
 *		tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
 *		rast_db: (Optional) Tensor containing image-space derivatives of barycentrics,
 *				 i.e., the second output tensor from `rasterize()`. Enables computing
 *				 image-space derivatives of attributes.
 *		diff_attrs: (Optional) List of attribute indices for which image-space
 *					derivatives are to be computed. Special value 'all' is equivalent
 *					to list [0, 1, ..., num_attributes - 1].
 *
 *	Returns:
 *		A tuple of two tensors. The first output tensor contains interpolated
 *		attributes and has shape [minibatch_size, height, width, num_attributes].
 *		If `rast_db` and `diff_attrs` were specified, the second output tensor contains
 *		the image-space derivatives of the selected attributes and has shape
 *		[minibatch_size, height, width, 2 * len(diff_attrs)]. The derivatives of the
 *		first selected attribute A will be on channels 0 and 1 as (dA/dX, dA/dY), etc.
 *		Otherwise, the second output tensor will be an empty tensor with shape
 *		[minibatch_size, height, width, 0].
 */
inline void interpolate(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db = torch::Tensor(), bool diff_attrs_all = false, std::vector<int>* diff_attrs_list = nullptr)
{
	if (diff_attrs_list)
	{
		_interpolate_func_da::apply(attr, rast, tri, rast_db, diff_attrs_all, *diff_attrs_list);
	}
	else
	{
		_interpolate_func::apply(attr, rast, tri);
	}
}

//----------------------------------------------------------------------------
// Texture
//----------------------------------------------------------------------------

enum class NVDRTextureFilterMode
{
	Auto,
	Nearest,
	Linear,
	LinearMipmapNearest,
	LinearMipmapLinear
};

enum class NVDRTextureBoundaryMode
{
	Wrap,
	Clamp,
	Zero,
	Cube
};

// Linear-mipmap-linear and linear-mipmap-nearest: Mipmaps enabled.
class _texture_func_mip : public torch::autograd::Function<_texture_func_mip>
{
public:
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, NVDRTextureFilterMode filterMode, 
		torch::Tensor tex, torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias,
		TextureMipWrapper mipWrapper,
		int filter_mode_enum, int boundary_mode_enum,
		torch::autograd::variable_list mip_stack)
	{
		torch::Tensor out = texture_fwd_mip(tex, uv, uv_da, mip_level_bias, mipWrapper, mip_stack, filter_mode_enum, boundary_mode_enum);

		torch::autograd::variable_list save = { tex, uv, uv_da, mip_level_bias };
		
		for (auto& e : mip_stack)
			save.push_back(e);

		ctx->save_for_backward(save);

		ctx->saved_data["filter_mode"] = (int)filterMode;
		ctx->saved_data["mip_wrapper.mip"] = mipWrapper.mip;
		ctx->saved_data["mip_wrapper.max_mip_level"] = mipWrapper.max_mip_level;
		ctx->saved_data["mip_wrapper.texture_size"] = mipWrapper.texture_size;
		ctx->saved_data["mip_wrapper.cube_mode"] = mipWrapper.cube_mode;
		ctx->saved_data["filter_mode_enum"] = filter_mode_enum;
		ctx->saved_data["boundary_mode_enum"] = boundary_mode_enum;

		return { out };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];

		torch::Tensor tex = ctx->get_saved_variables()[0];
		torch::Tensor uv = ctx->get_saved_variables()[1];
		torch::Tensor uv_da = ctx->get_saved_variables()[2];
		torch::Tensor mip_level_bias = ctx->get_saved_variables()[3];

		torch::autograd::variable_list mip_stack;
		for (size_t i = 4; i < ctx->get_saved_variables().size(); i++)
		{
			mip_stack.push_back(ctx->get_saved_variables()[i]);
		}

		NVDRTextureFilterMode filterMode = (NVDRTextureFilterMode)ctx->saved_data["filter_mode"].toInt();

		TextureMipWrapper mipWrapper;
		mipWrapper.mip = ctx->saved_data["mip_wrapper.mip"].toTensor();
		mipWrapper.max_mip_level = (int)ctx->saved_data["mip_wrapper.max_mip_level"].toInt();
		mipWrapper.texture_size = ctx->saved_data["mip_wrapper.texture_size"].toIntVector();
		mipWrapper.cube_mode = ctx->saved_data["mip_wrapper.cube_mode"].toBool();

		int filter_mode_enum = (int)ctx->saved_data["filter_mode_enum"].toInt();
		int boundary_mode_enum = (int)ctx->saved_data["boundary_mode_enum"].toInt();

		if (filterMode == NVDRTextureFilterMode::LinearMipmapLinear)
		{
			auto result = texture_grad_linear_mipmap_linear(tex, uv, dy, uv_da, mip_level_bias, mipWrapper, mip_stack, filter_mode_enum, boundary_mode_enum);

			torch::Tensor g_tex = std::get<0>(result);
			torch::Tensor g_uv = std::get<1>(result);
			torch::Tensor g_uv_da = std::get<2>(result);
			torch::Tensor g_mip_level_bias = std::get<3>(result);
			torch::autograd::variable_list g_mip_stack = std::get<4>(result);

			torch::autograd::variable_list r = { torch::Tensor(), g_tex, g_uv, g_uv_da, g_mip_level_bias, torch::Tensor(), torch::Tensor(), torch::Tensor() };
			for (auto& e : g_mip_stack)
				r.push_back(e);
			return r;
		}
		else
		{
			auto result = texture_grad_linear_mipmap_nearest(tex, uv, dy, uv_da, mip_level_bias, mipWrapper, mip_stack, filter_mode_enum, boundary_mode_enum);
			
			torch::Tensor g_tex = std::get<0>(result);
			torch::Tensor g_uv = std::get<1>(result);
			torch::autograd::variable_list g_mip_stack = std::get<2>(result);

			torch::autograd::variable_list r = { torch::Tensor(), g_tex, g_uv, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor() };
			for (auto& e : g_mip_stack)
				r.push_back(e);
			return r;
		}
	}
};

// Linear and nearest: Mipmaps disabled
class _texture_func : public torch::autograd::Function<_texture_func>
{
public:
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, NVDRTextureFilterMode filterMode,
		torch::Tensor tex, torch::Tensor uv, 
		int filter_mode_enum, int boundary_mode_enum)
	{
		torch::Tensor out = texture_fwd(tex, uv, filter_mode_enum, boundary_mode_enum);

		ctx->save_for_backward({ tex, uv });

		ctx->saved_data["filter_mode"] = (int)filterMode;
		ctx->saved_data["filter_mode_enum"] = filter_mode_enum;
		ctx->saved_data["boundary_mode_enum"] = boundary_mode_enum;

		return { out };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];

		torch::Tensor tex = ctx->get_saved_variables()[0];
		torch::Tensor uv = ctx->get_saved_variables()[1];

		torch::autograd::variable_list mip_stack;
		for (size_t i = 2; i < ctx->get_saved_variables().size(); i++)
		{
			mip_stack.push_back(ctx->get_saved_variables()[i]);
		}

		NVDRTextureFilterMode filterMode = (NVDRTextureFilterMode)ctx->saved_data["filter_mode"].toInt();
		int filter_mode_enum = ctx->saved_data["filter_mode_enum"].toInt();
		int boundary_mode_enum = ctx->saved_data["boundary_mode_enum"].toInt();

		if (filterMode == NVDRTextureFilterMode::Linear)
		{
			auto result = texture_grad_linear(tex, uv, dy, filter_mode_enum, boundary_mode_enum);

			torch::Tensor g_tex = std::get<0>(result);
			torch::Tensor g_uv = std::get<1>(result);

			return { torch::Tensor(), g_tex, g_uv, torch::Tensor(), torch::Tensor() };
		}
		else
		{
			torch::Tensor g_tex = texture_grad_nearest(tex, uv, dy, filter_mode_enum, boundary_mode_enum);
			return { torch::Tensor(), g_tex, torch::Tensor(), torch::Tensor(), torch::Tensor() };
		}
	}
};

/*
 * Perform texture sampling.
 *
 * All input tensors must be contiguous and reside in GPU memory. The output tensor
 * will be contiguous and reside in GPU memory.
 *
 * Args:
 * 	tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
 * 		 [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
 * 		 must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
 * 		 tex_width and tex_height are equal. Note that `boundary_mode` must also be set
 * 		 to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
 * 	uv: Tensor containing per-pixel texture coordinates. When sampling a 2D texture,
 * 		must have shape [minibatch_size, height, width, 2]. When sampling a cube map
 * 		texture, must have shape [minibatch_size, height, width, 3].
 * 	uv_da: (Optional) Tensor containing image-space derivatives of texture coordinates.
 * 		   Must have same shape as `uv` except for the last dimension that is to be twice
 * 		   as long.
 * 	mip_level_bias: (Optional) Per-pixel bias for mip level selection. If `uv_da` is omitted,
 * 					determines mip level directly. Must have shape [minibatch_size, height, width].
 * 	mip: (Optional) Preconstructed mipmap stack from a `texture_construct_mip()` call, or a list
 * 					of tensors specifying a custom mipmap stack. When specifying a custom mipmap stack,
 * 					the tensors in the list must follow the same format as `tex` except for width and
 * 					height that must follow the usual rules for mipmap sizes. The base level texture
 * 					is still supplied in `tex` and must not be included in the list. Gradients of a
 * 					custom mipmap stack are not automatically propagated to base texture but the mipmap
 * 					tensors will receive gradients of their own. If a mipmap stack is not specified
 * 					but the chosen filter mode requires it, the mipmap stack is constructed internally
 * 					and discarded afterwards.
 * 	filter_mode: Texture filtering mode to be used. Valid values are 'auto', 'nearest',
 * 				 'linear', 'linear-mipmap-nearest', and 'linear-mipmap-linear'. Mode 'auto'
 * 				 selects 'linear' if neither `uv_da` or `mip_level_bias` is specified, and
 * 				 'linear-mipmap-linear' when at least one of them is specified, these being
 * 				 the highest-quality modes possible depending on the availability of the
 * 				 image-space derivatives of the texture coordinates or direct mip level information.
 * 	boundary_mode: Valid values are 'wrap', 'clamp', 'zero', and 'cube'. If `tex` defines a
 * 				   cube map, this must be set to 'cube'. The default mode 'wrap' takes fractional
 * 				   part of texture coordinates. Mode 'clamp' clamps texture coordinates to the
 * 				   centers of the boundary texels. Mode 'zero' virtually extends the texture with
 * 				   all-zero values in all directions.
 * 	max_mip_level: If specified, limits the number of mipmaps constructed and used in mipmap-based
 * 				   filter modes.
 *
 * Returns:
 * 	A tensor containing the results of the texture sampling with shape
 * 	[minibatch_size, height, width, tex_channels]. Cube map fetches with invalid uv coordinates
 * 	(e.g., zero vectors) output all zeros and do not propagate gradients.
 */
inline torch::autograd::variable_list texture(torch::Tensor tex, torch::Tensor uv,
	torch::Tensor uv_da = torch::Tensor(), torch::Tensor mip_level_bias = torch::Tensor(), const TextureMipWrapper* mip = nullptr,
	NVDRTextureFilterMode filter_mode = NVDRTextureFilterMode::Auto, NVDRTextureBoundaryMode boundary_mode = NVDRTextureBoundaryMode::Wrap, int maxMipLevel = -1)
{
	// Default filter mode.
	if (filter_mode == NVDRTextureFilterMode::Auto)
	{
		if (uv_da.size(0) || mip_level_bias.size(0))
			filter_mode = NVDRTextureFilterMode::LinearMipmapLinear;
		else
			filter_mode = NVDRTextureFilterMode::Linear;
	}

	assert(max_mip_level == -1 || max_mip_level >= 0);

	// If mipping disabled via max level=0, we may as well use simpler filtering internally.
	if (maxMipLevel == 0 && (filter_mode == NVDRTextureFilterMode::LinearMipmapNearest || filter_mode == NVDRTextureFilterMode::LinearMipmapLinear))
		filter_mode = NVDRTextureFilterMode::Linear;

	// Convert filter mode to internal enumeration
	int filter_mode_enum = 0;
	switch (filter_mode)
	{
	case NVDRTextureFilterMode::Linear:
		filter_mode_enum = 1;
		break;
	case  NVDRTextureFilterMode::LinearMipmapNearest:
		filter_mode_enum = 2;
		break;
	case  NVDRTextureFilterMode::LinearMipmapLinear:
		filter_mode_enum = 3;
		break;
	}

	// Convert boundary mode to internal enumeration
	int boundary_mode_enum = 0;
	switch (boundary_mode)
	{
	case NVDRTextureBoundaryMode::Wrap:
		boundary_mode_enum = 1;
		break;
	case NVDRTextureBoundaryMode::Clamp:
		boundary_mode_enum = 2;
		break;
	case NVDRTextureBoundaryMode::Zero:
		boundary_mode_enum = 3;
		break;
	}

	// Construct a mipmap if necessary
	TextureMipWrapper mip_wrapper;
	torch::autograd::variable_list mip_stack;

	if (mip)
	{
		mip_wrapper = *mip;
	}
	else
	{
		mip_wrapper = texture_construct_mip(tex, maxMipLevel, boundary_mode == NVDRTextureBoundaryMode::Cube);
	}

	if (filter_mode == NVDRTextureFilterMode::LinearMipmapLinear || filter_mode == NVDRTextureFilterMode::LinearMipmapNearest)
	{
		return _texture_func_mip::apply(filter_mode, tex, uv, uv_da, mip_level_bias, mip_wrapper, filter_mode_enum, boundary_mode_enum, mip_stack);
	}
	else
	{
		return _texture_func::apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum);
	}
}

// ----------------------------------------------------------------------------
//  Antialias.
// ----------------------------------------------------------------------------

class _antialias_func : public torch::autograd::Function<_antialias_func>
{
public:
	static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
		torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri,
		TopologyHashWrapper topology_hash, float pos_gradient_boost)
	{
		auto result = antialias_fwd(color, rast, pos, tri, topology_hash);

		torch::Tensor out = std::get<0>(result);
		torch::Tensor work_buffer = std::get<1>(result);

		ctx->save_for_backward({ color, rast, pos, tri });
		ctx->saved_data["pos_gradient_boost"] = (double)pos_gradient_boost;
		ctx->saved_data["work_buffer"] = work_buffer;

		return { out };
	}

	static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		torch::Tensor dy = grad_output[0];

		torch::Tensor color = ctx->get_saved_variables()[0];
		torch::Tensor rast = ctx->get_saved_variables()[1];
		torch::Tensor pos = ctx->get_saved_variables()[2];
		torch::Tensor tri = ctx->get_saved_variables()[3];

		float pos_gradient_boost = (float)ctx->saved_data["pos_gradient_boost"].toDouble();
		torch::Tensor work_buffer = ctx->saved_data["work_buffer"].toTensor();

		auto result = antialias_grad(color, rast, pos, tri, dy, work_buffer);

		torch::Tensor g_color = std::get<0>(result);
		torch::Tensor g_pos = std::get<1>(result);

		if (pos_gradient_boost != 1.0f)
			g_pos = g_pos * pos_gradient_boost;

		return { g_color, torch::Tensor(), g_pos, torch::Tensor(), torch::Tensor() };
	}
};

/*
 * Perform antialiasing.
 *
 *	All input tensors must be contiguous and reside in GPU memory. The output tensor
 *	will be contiguous and reside in GPU memory.
 *
 *	Note that silhouette edge determination is based on vertex indices in the triangle
 *	tensor. For it to work properly, a vertex belonging to multiple triangles must be
 *	referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always
 *	classify the adjacent edges as silhouette edges, which leads to bad performance and
 *	potentially incorrect gradients. If you are unsure whether your data is good, check
 *	which pixels are modified by the antialias operation and compare to the example in the
 *	documentation.
 *
 *	Args:
 *		color: Input image to antialias with shape [minibatch_size, height, width, num_channels].
 *		rast: Main output tensor from `rasterize()`.
 *		pos: Vertex position tensor used in the rasterization operation.
 *		tri: Triangle tensor used in the rasterization operation.
 *		topology_hash: (Optional) Preconstructed topology hash for the triangle tensor. If not
 *					   specified, the topology hash is constructed internally and discarded afterwards.
 *		pos_gradient_boost: (Optional) Multiplier for gradients propagated to `pos`.
 *
 *	Returns:
 *		A tensor containing the antialiased image with the same shape as `color` input tensor.
 */
static torch::autograd::variable_list antialias(torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri,
	TopologyHashWrapper* _topology_hash = nullptr, float pos_gradient_boost = 1.0f)
{
	// Construct topology hash unless provided by user.
	TopologyHashWrapper topology_hash;
	if (_topology_hash)
		topology_hash = antialias_construct_topology_hash(tri);

	return _antialias_func::apply(color, rast, pos, tri, topology_hash, pos_gradient_boost);
}

