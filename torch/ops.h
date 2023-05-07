#pragma once
#include "torch_types.h"

#define NDR_API
/*#ifdef NDR_EXPORT
#define NDR_API __declspec(dllimport)
#else
#define NDR_API __declspec(dllexport)
#endif
*/

#pragma comment(lib, "torch.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda_cpp.lib")
#pragma comment(lib, "torch_cuda_cu.lib")
#pragma comment(lib, "c10.lib")
#pragma comment(lib, "c10_cuda.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "setgpu.lib")
#pragma comment(lib, "cudart_static.lib")

int GetCurrentCUDADeviceIndex(torch::Device device);

class DepthPeeler;

class NDR_API RasterizeBaseContext
{
public:
	virtual bool isGLContext() const = 0;

	virtual ~RasterizeBaseContext() { }

protected:
	RasterizeBaseContext(bool output_db)
		: m_output_db(output_db)
	{ }

public:
	bool m_output_db;
	DepthPeeler* m_active_depth_peeler = nullptr;
};

class NDR_API RasterizeCudaContext : public RasterizeBaseContext
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

class NDR_API RasterizeGLContext : public RasterizeBaseContext
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
NDR_API torch::autograd::variable_list rasterize(RasterizeBaseContext* glctx, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges = torch::Tensor(), bool grad_db = true);

// ----------------------------------------------------------------------------
//  Depth peeler context manager for rasterizing multiple depth layers.
// ----------------------------------------------------------------------------
class NDR_API DepthPeeler
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

	torch::autograd::variable_list rasterize_next_layer();

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
NDR_API torch::autograd::variable_list interpolate(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db = torch::Tensor(), bool diff_attrs_all = false, std::vector<int>* diff_attrs_list = nullptr);

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
NDR_API torch::autograd::variable_list texture(torch::Tensor tex, torch::Tensor uv,
	torch::Tensor uv_da = torch::Tensor(), torch::Tensor mip_level_bias = torch::Tensor(), const TextureMipWrapper* mip = nullptr,
	NVDRTextureFilterMode filter_mode = NVDRTextureFilterMode::Auto, NVDRTextureBoundaryMode boundary_mode = NVDRTextureBoundaryMode::Wrap, int maxMipLevel = -1);

// ----------------------------------------------------------------------------
//  Antialias.
// ----------------------------------------------------------------------------

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
NDR_API torch::autograd::variable_list antialias(torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri,
	TopologyHashWrapper* _topology_hash = nullptr, float pos_gradient_boost = 1.0f);
