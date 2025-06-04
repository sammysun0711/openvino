// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "intel_gpu/runtime/error_handler.hpp"
#include "register.hpp"
#include "registry/implementation_map.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_ext.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

namespace cldnn {
namespace ocl {
static bool enable_p2p_debug = getenv("OV_GPU_P2P_DEBUG");
static std::mutex debug_mutex;
class Timer {
public:
    Timer(size_t rank) : my_rank(rank) {
        if (enable_p2p_debug) {
            mark_timers = true;
        }
    }
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::micro>;

    void measure_start(std::string msg) {
        if (this->mark_timers) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "[rank:] " << my_rank << " stage : " << msg << " TIME START " << std::endl;
        }
        start_time = Clock::now();
    }

    void measure_end(std::string msg) {
        end_time = Clock::now();
        if (this->mark_timers) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << "[rank:] " << my_rank << " stage : " << msg << " TIME END " << std::endl;
            std::cout << "[rank:] " << my_rank << " stage : " << msg << " COST " << get().count() << std::endl;
        }
    }

    Duration get() const {
        if (end_time <= start_time) {
            return std::chrono::nanoseconds(1);
        }
        return end_time - start_time;
    }

private:
    bool mark_timers = false;
    Clock::time_point start_time;
    Clock::time_point end_time;
    size_t my_rank;
};

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
#define CL_MEM_DEVICE_ID_INTEL         0x10011

class tensor_concat_memory {
public:
    tensor_concat_memory() : buf(nullptr), width(0), height(0), type(ov::element::f16) {}
    tensor_concat_memory(cl_mem _buf, size_t _w, size_t _h, size_t _stride, ov::element::Type _type)
        : buf(_buf),
          width(_w),
          height(_h),
          stride(_stride),
          type(_type) {}
    tensor_concat_memory(tensor_concat_memory& other) {
        buf = other.buf;
        width = other.width;
        height = other.height;
        stride = other.stride;
        type = other.type;
    }
    bool operator==(const tensor_concat_memory& other) const {
        return width == other.height && height == other.height && stride == other.stride;
    }

    void get_mem_info() const {
        size_t data_size = 0;
        cl::detail::errHandler(clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL),
                               "cl_get_mem_obj_info");
        GPU_DEBUG_TRACE_DETAIL << "width = " << width << ", height = " << height << ", stride = " << stride
                               << ", type = " << type.to_string() << " -- actual_size = " << data_size << std::endl;
    }

    cl_mem buf;
    size_t width;
    size_t height;
    size_t stride;
    ov::element::Type type;
};

class simple_tensor_concat {
public:
    simple_tensor_concat() {}
    ~simple_tensor_concat() {}

    bool validate(std::shared_ptr<tensor_concat_memory>& src, std::shared_ptr<tensor_concat_memory>& dst) {
        if (!src->buf)
            return false;
        if (src->type != dst->type)
            return false;

        if (!dst->buf)
            return false;

        concat_mode = -1;
        if (src->width == dst->width) {
            // Vertical concat
            concat_mode = 0;
        } else if (src->height <= dst->height) {  // fake alignment issue
            // Horizontal concat
            concat_mode = 1;
        } else {
            return false;
        }
        return true;
    }

    cldnn::event::ptr concat(cldnn::stream& stream,
                             std::shared_ptr<tensor_concat_memory>& src,
                             std::shared_ptr<tensor_concat_memory>& dst,
                             size_t w_rank,
                             bool blocked = true) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();

        if (!validate(src, dst)) {
            std::lock_guard<std::mutex> lock(debug_mutex);
            get_mem_info(src, dst);
            GPU_DEBUG_TRACE_DETAIL << "simple_tensor_concat::validate failed due to src/dst mismatch.";
        }

        size_t src_rec[3] = {0, 0, 0};
        size_t dst_rec[3] = {0, 0, 0};
        size_t rect[3] = {src->width, src->height, 1};
        cl_event event;
        cldnn::event::ptr sync_event = nullptr;
        if (concat_mode == 0) {
            // Vertical concat
            dst_rec[1] = src->height * w_rank;
            cl::detail::errHandler(clEnqueueCopyBufferRect(queue,
                                                           src->buf,
                                                           dst->buf,
                                                           src_rec,
                                                           dst_rec,
                                                           rect,
                                                           src->stride,
                                                           src->height * src->stride,
                                                           dst->stride,
                                                           dst->stride * dst->width,
                                                           0,
                                                           nullptr,
                                                           &event),
                                   "clEnqueueCopyBufferRect");
        } else if (concat_mode == 1) {
            // Horizontal concat
            dst_rec[0] = src->width * w_rank;
            cl::detail::errHandler(clEnqueueCopyBufferRect(queue,
                                                           src->buf,
                                                           dst->buf,
                                                           src_rec,
                                                           dst_rec,
                                                           rect,
                                                           src->stride,
                                                           src->height * src->stride,
                                                           dst->stride,
                                                           dst->stride * dst->width,
                                                           0,
                                                           nullptr,
                                                           &event),
                                   "clEnqueueCopyBufferRect");
        } else {
            OPENVINO_THROW("tensor_concat failed: incorrect concat mode!");
        }
        if (blocked) {
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
        } else {
            sync_event = ocl_stream.create_event(cl::Event(event));
        }
        return sync_event;
    }

    void get_mem_info(const std::shared_ptr<tensor_concat_memory>& src,
                      const std::shared_ptr<tensor_concat_memory>& dst) {
        src->get_mem_info();
        dst->get_mem_info();
    }

private:
    // 0 - vertical concat
    // 1 - horizontal concat
    int concat_mode;
};

class gpu_p2p_helper {
public:
    gpu_p2p_helper() {}
    ~gpu_p2p_helper() {}

    cl_mem map_remote_mem(cl_context context, cl_device_id device_handle, cl_mem clbuf, size_t size) {
        cl_int err;
        uint64_t fd;
        err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(fd), &fd, NULL);
        if (err < 0) {
            GPU_DEBUG_TRACE_DETAIL << "FAILED to derive handle from mem obj " << clbuf << std::endl;;
        }
        // std::cout << "finished to derive handle from " << clbuf << "handle is " << fd << std::endl;
        //  Create extMemBuffer of type cl_mem from fd.
        cl_mem_properties_intel extMemProperties[] = {(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
                                                (cl_mem_properties)fd,
                                                CL_MEM_DEVICE_ID_INTEL,
                                                (cl_mem_properties_intel)device_handle,
                                                0};
        cl_mem extMemBuffer = clCreateBufferWithProperties(context, extMemProperties, 0, size, NULL, &err);
        if (err < 0) {
            GPU_DEBUG_TRACE_DETAIL << "FAILED to import buffer from mem " << clbuf << ", fd as " << fd << std::endl;
            OPENVINO_ASSERT(false,
                            "clCreateBufferWithProperties failed, clbuf = %p, fd = %ld, size = %ld, new_cl_mem = %p\n",
                            clbuf,
                            fd,
                            size,
                            extMemBuffer);
        }

        return extMemBuffer;
    }

    void destory_remote_mem(cl_mem clbuf) {
        clReleaseMemObject(clbuf);
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

    void remote_copy(cldnn::stream& stream, cl_mem src, cl_mem dst, size_t size) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 0, NULL, &ret);
        clWaitForEvents(1, &ret);  // blocked copy
        clReleaseEvent(ret);
        return;
    }

    std::shared_ptr<tensor_concat_memory> create_concat_mem(cl_mem src, cldnn::layout src_layout) {
        ov::element::Type element_type = src_layout.data_type;
        auto element_size = element_type.size();
        auto src_shape = src_layout.get_shape();
        auto src_width = src_shape[-1] * element_size;
        auto src_stride = src_shape[-1] * element_size;  // No pad
        auto src_height = ov::shape_size(src_shape) / src_shape[-1];
        return std::make_shared<tensor_concat_memory>(src, src_width, src_height, src_stride, element_type);
    }

    cldnn::event::ptr remote_copy_rect(cldnn::stream& stream,
                                       cl_mem src,
                                       cldnn::layout src_layout,
                                       cl_mem dst,
                                       cldnn::layout dst_layout,
                                       size_t w_rank,
                                       bool blocked) {
        auto concat = std::make_shared<simple_tensor_concat>();
        auto mem_src = create_concat_mem(src, src_layout);
        auto mem_dst = create_concat_mem(dst, dst_layout);
        auto ret = concat->concat(stream, mem_src, mem_dst, w_rank, blocked);
        return ret;
    }
};

class simple_tensor_add {
public:
    simple_tensor_add() {}
    ~simple_tensor_add() {
        for (auto& item : kernels) {
            if (item.second)
                clReleaseKernel(item.second);
        }
        kernels.clear();
        if (program)
            clReleaseProgram(program);
    }

    typedef enum _kernel_data_type {
        e_type_fp16 = 0,
        e_type_int8 = 1,
        e_type_fp32 = 2,
    } kernel_data_type;

    kernel_data_type element_type_to_kernel_data_type(ov::element::Type_t element_type) {
        switch (element_type) {
        case ov::element::f16:
            return kernel_data_type::e_type_fp16;
        case ov::element::i8:
            return kernel_data_type::e_type_int8;
        case ov::element::f32:
            return kernel_data_type::e_type_fp32;
        default:
            OPENVINO_THROW("Error: unsupported element type for kernel adder - ",
                           ov::element::Type(element_type).to_string().c_str());
            break;
        }
        return kernel_data_type::e_type_int8;
    }

    cl_kernel create_kernel(cldnn::stream& stream, const char* kernel_code, const char* kernelName) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        GPU_DEBUG_TRACE_DETAIL << "create_kernel: name = " << kernelName;

        cl_uint knlcount = 1;
        const char* knlstrList[] = {kernel_code};
        size_t knlsizeList[] = {strlen(kernel_code)};

        cl_context context = ocl_stream.get_engine().get_cl_context().get();
        program = clCreateProgramWithSource(context, knlcount, knlstrList, knlsizeList, &err);

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0) {
            size_t logsize = 0;
            auto device = ocl_stream.get_engine().get_cl_device().get();
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            // CHECK_OCL_ERROR(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            GPU_DEBUG_TRACE_DETAIL << "clGetProgramBuildInfo failed: " << logbuf.data();
            // OPENVINO_ASSERT(err >= 0, "clGetProgramBuildInfo: ", logbuf.data());
        }
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        return kernel;
    }

    cl_kernel get_or_create_kernel_if_possible_sub(cldnn::stream& stream,
                                                   kernel_data_type type,
                                                   size_t offset) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = kernels.find(type);
        if (it != kernels.end()) {
            return it->second;
        }

#define ADD_OP_KERNEL_SOURCE_CODE(DATA_TYPE)                                                             \
    "kernel void tensor_add_kernel_" #DATA_TYPE " (const global " #DATA_TYPE " *src, global " #DATA_TYPE \
    " *dst, int offset) {"                                                     \
    "const int id = get_global_id(0);"                                                                   \
    "const int idx = id + offset;"                                                                       \
    "dst[idx] += src[id];"                                                                               \
    "}"
        if (type == kernel_data_type::e_type_fp16) {
            const char tensor_add_kernel_fp16[] = ADD_OP_KERNEL_SOURCE_CODE(half);
            const char kernel_name[] = "tensor_add_kernel_half";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp16, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_int8) {
            const char tensor_add_kernel_int8[] = ADD_OP_KERNEL_SOURCE_CODE(char);
            const char kernel_name[] = "tensor_add_kernel_char";
            kernels[type] = create_kernel(stream, tensor_add_kernel_int8, kernel_name);
            return kernels[type];
        } else if (type == kernel_data_type::e_type_fp32) {
            const char tensor_add_kernel_fp32[] = ADD_OP_KERNEL_SOURCE_CODE(float);
            const char kernel_name[] = "tensor_add_kernel_float";
            kernels[type] = create_kernel(stream, tensor_add_kernel_fp32, kernel_name);
            return kernels[type];
        } else {
            OPENVINO_THROW("error: unsupported adder kernel data type ", static_cast<int>(type));
        }
#undef ADD_OP_KERNEL_SOURCE_CODE
        return kernels[type];
    }

    event::ptr tensor_add_sub(cldnn::stream& stream,
                              cl_mem src,
                              cl_mem dst,
                              size_t element_count,
                              kernel_data_type data_type,
                              size_t offset) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            GPU_DEBUG_TRACE_DETAIL << "tensor_add: invalid arguments!";
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        cl_kernel kernel = get_or_create_kernel_if_possible_sub(stream,
                                                                data_type,
                                                                static_cast<int>(offset));

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);

        err = clSetKernelArg(kernel, 2, sizeof(size_t), &offset);

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();
        cl_event ret;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &ret);
        if (err < 0)
            cl::detail::errHandler(err, "set kernel args and enqueue ND kernel");
        return ocl_stream.create_event(cl::Event(ret));
    }

    void tensor_add_sub_1(cldnn::stream& stream,
                          cl_mem src,
                          cl_mem dst,
                          size_t element_count,
                          kernel_data_type data_type,
                          size_t offset,
                          cl_uint num_events_in_wait_list,
                          const cl_event* event_wait_list,
                          cl_event* event) {
        cl_int err;
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        if (src == nullptr || dst == nullptr) {
            GPU_DEBUG_TRACE_DETAIL << "tensor_add: invalid arguments!";
        }
        OPENVINO_ASSERT(src != nullptr && dst != nullptr, "tensor_add: invalid arguments!");

        cl_kernel kernel = get_or_create_kernel_if_possible_sub(stream, data_type, static_cast<int>(offset));

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);

        err = clSetKernelArg(kernel, 2, sizeof(size_t), &offset);

        size_t global_size[] = {element_count};
        auto queue = ocl_stream.get_cl_queue().get();

        err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,
                                     nullptr,
                                     global_size,
                                     nullptr,
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     event);

        if (err != CL_SUCCESS) {
            OPENVINO_THROW("enqueue add event failed!");
        }
    }

    void finish(cldnn::stream& stream) {
        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();
        clFinish(queue);
    }

private:
    cl_program program;
    std::mutex mutex;
    std::map<kernel_data_type, cl_kernel> kernels;
};

static gpu_p2p_helper& get_p2p_instance() {
    static gpu_p2p_helper gpu_p2p_instance;
    return gpu_p2p_instance;
}

static simple_tensor_add& get_adder_instance(size_t idx) {
    static simple_tensor_add adder_instance[4];
    return adder_instance[idx];
}

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    ~sync_tensor_impl() {
        for (auto& mem : all_gather_remote_dst) {
            if (mem) {
                release_remote_mems(static_cast<cl_mem>(mem));
            }
        }
        all_gather_remote_dst.clear();
    }

    explicit sync_tensor_impl(const sync_tensor_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<sync_tensor>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    bool update_internal_buffer(sync_tensor_inst& instance,
                                cl_context context,
                                cl_device_id handle,
                                std::vector<cldnn::memory::ptr>& bufs,
                                cldnn::layout& last_layout,
                                cldnn::layout& layout,
                                size_t w_size,
                                size_t w_rank) {
        auto& engine = instance.get_network().get_engine();
        auto& ocl_engine = downcast<ocl::ocl_engine>(engine);
        size_t required_size = layout.bytes_count();
        bool allocated = false;

        auto need_realloc = [&](size_t idx) {
            if (idx != 1)
                return false;

            if (bufs[idx] == nullptr || last_layout.bytes_count() == 0)
                return true;

            if (bufs[idx]->size() * w_size < required_size)
                return true;

            // Batch has been changed to smaller, need reallocate to decrease memory.
            auto last_batch = last_layout.batch() * last_layout.feature();
            auto batch = layout.batch() * layout.feature();
            if (last_batch > batch)
                return true;
            return false;
        };
        for (size_t i = 0; i < w_size; i++) {
            if (!need_realloc(i)) {
                continue;
            }
            bufs[i] = nullptr;
            auto width = layout.get_shape()[-1];
            auto sub_width = width / w_size;
            if (sub_width * w_size != width)
                GPU_DEBUG_TRACE_DETAIL << "[Warning] the shape of FC output has ODD number!!!";
            auto sub_layout = layout;
            auto sub_shape = layout.get_shape();
            sub_shape[-1] = sub_width;
            sub_layout.set_partial_shape(sub_shape);

            // Create extMemBuffer of type cl_mem from fd.
            cl_mem_properties_intel extMemProperties[] = {
                CL_MEM_FLAGS,
                CL_MEM_READ_WRITE | CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL,
                CL_MEM_DEVICE_ID_INTEL,
                (cl_mem_properties_intel)handle,
                0,
            };
            auto local_mem =
                clCreateBufferWithProperties(context, extMemProperties, 0, sub_layout.bytes_count(), NULL, NULL);
            bufs[i] = std::make_shared<ocl::gpu_buffer>(&ocl_engine, sub_layout, cl::Buffer(local_mem, true), nullptr);
            allocated = true;
        }
        return allocated;
    }

    void release_remote_mems(cl_mem remote_mems) {
        if (remote_mems) {
            auto error = clReleaseMemObject(remote_mems);
            if (error != CL_SUCCESS) {
                GPU_DEBUG_TRACE_DETAIL << "[Failed]][Release] cl buf: " << remote_mems << std::endl;
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& local_stream = instance.get_network().get_stream();
        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        gpu_p2p_helper& gpu_p2p_instance = get_p2p_instance();
        auto& ocl_stream = downcast<ocl::ocl_stream>(local_stream);
        auto local_queue = ocl_stream.get_cl_queue().get();                        // raw cl queue
        auto local_context = ocl_stream.get_engine().get_cl_context().get();       // raw cl context
        auto local_device_handle = ocl_stream.get_engine().get_cl_device().get();  // raw cl device
        auto dst_idx = (w_rank + 1) % w_size;                                      // peer for p2p of current rank in ring all_reduce
        Timer timer(w_rank);
        auto is_all_reduce = instance.get_impl_params()->all_reduce == true;
        if (!is_all_reduce && all_gather_remote_dst.size() == 0) {
            all_gather_remote_dst.assign(w_size, nullptr);
        }
        instance.sync_wait_times = timer.get().count();
        // use ring all-reduce solution to implement
        timer.measure_start(std::string("prepare for sync for mem manager"));
        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->updated_flag = false;
            sub_mem_mgr->step1_copy_done.store(0);
            sub_mem_mgr->step2_add_done.store(0);
            sub_mem_mgr->step3_concat_copy_done.store(0);
            sub_mem_mgr->step4_concat_copy_done.store(0);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                    sub_mem_mgr->_memorys_table[id][i].all_gather_flag = false;
                    sub_mem_mgr->_memorys_table[id][i].all_gather_copy_flag = false;
                    for (size_t j = 0; j < w_size; j++) {
                        sub_mem_mgr->_memorys_table[id][i].events[j] = nullptr;
                        sub_mem_mgr->_memorys_table[id][i].recv_flag[j] = false;
                    }
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }
        timer.measure_end(std::string("prepare for sync for mem manager"));
        instance.sub_mem_manager_sync_times = timer.get().count();
        timer.measure_start(std::string("prepare for sync for local memory allocation"));

        auto p2p_src_layout = instance.get_output_layout(0);
        bool need_update_remote_mems = false;
        if (is_all_reduce) {
            OPENVINO_ASSERT(1 == instance.get_output_memorys().size(), "All reduce only has one output!");
            sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0] = instance.get_output_memorys()[0];
            sub_mem_mgr->_memorys_table[id][w_rank].output = instance.get_output_memorys()[0];
            // Allocate or reuse buffer for P2P target, same shape with output[0]
            p2p_src_layout = instance.get_output_layout(0);
            need_update_remote_mems = update_internal_buffer(instance,
                                                             local_context,
                                                             local_device_handle,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs,
                                                             sub_mem_mgr->_memorys_table[id][w_rank].layout,
                                                             p2p_src_layout,
                                                             w_size,
                                                             w_rank);
        } else {
            OPENVINO_ASSERT(2 == instance.get_output_memorys().size(),
                            "All gather need additional buffer for concat result!");
            // All gather doesn't need intermediate buffer at all.
            p2p_src_layout = instance.get_output_layout(1);
            auto tmp =
                std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.get_output_memorys()[0])->get_buffer().get();
            if (tmp != all_gather_current_dst) {
                need_update_remote_mems = true;
                all_gather_current_dst = tmp;
            }
        }

        sub_mem_mgr->_memorys_table[id][w_rank].recv_flag[0] = true;
        sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        timer.measure_end(std::string("prepare for sync for local memory allocation"));
        instance.all_reduce_local_mem_alloc_times = timer.get().count();
        std::vector<cldnn::event::ptr> sync_events;
        if (is_all_reduce) {
            timer.measure_start(std::string("prepare remote memory mapping"));
            // wait for peer memory ready
            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].recv_flag[0] == true)
                        wait_all_ouput_ready++;
                }
                if (wait_all_ouput_ready == w_size) {
                    break;
                }
            }
            auto split_parts = [](int len, int n) {
                int average = len / n;
                std::vector<size_t> parts(n, average);
                parts.back() = len - average * (n - 1);
                return parts;
            };

            auto output_layout = instance.get_output_layout(0);
            ov::element::Type output_element_type = output_layout.data_type;
            auto output_element_size = output_element_type.size();
            auto slice_pitch = output_layout.bytes_count();
            auto chunk_size = split_parts(slice_pitch, w_size);
            auto src_width = output_layout.get_shape()[-1];
            auto src_height = output_layout.count() / src_width;
            auto dst_height = src_height;
            // Prepare CL memory mapping for P2P copying next
            {
                cl_mem dst_cl_buf = nullptr;
                cldnn::memory::ptr dst_mem = sub_mem_mgr->_memorys_table[id][dst_idx].recv_bufs[1];
                size_t data_size = dst_mem->size();
                auto dst_cl_buf_remote = std::dynamic_pointer_cast<const ocl::gpu_buffer>(dst_mem)->get_buffer().get();
                dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);
                // The mapped remote cl_mem will hold the original cl_mem, it should be released if the original cl_mem
                // has been
                // released, else it will cause gpu memory leak.
                if (need_update_remote_mems) {
                    if (enable_p2p_debug) {
                        GPU_DEBUG_TRACE_DETAIL << "release_remote_mems: old_layout = "
                                               << sub_mem_mgr->_memorys_table[id][w_rank].layout.to_short_string()
                                               << ", new_layout = " << p2p_src_layout.to_short_string();
                    }
                }
                if (need_update_remote_mems || dst_cl_buf == nullptr) {
                    if (dst_cl_buf) {
                        //release_remote_mems(dst_cl_buf);
                    }
                    dst_cl_buf = gpu_p2p_instance.map_remote_mem(local_context,
                                                                 local_device_handle,
                                                                 dst_cl_buf_remote,
                                                                 data_size);
                    sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0] = dst_cl_buf;
                }
            }
            timer.measure_end(std::string("prepare for remote memory mapping"));
            instance.all_reduce_remote_mem_mapping_times = timer.get().count();
            timer.measure_start(std::string("scatter stage for Ring all-reduce"));
            // scatter stage for Ring all-reduce
            cl_int ret;
            for (int32_t step = 0; step < static_cast<int>(w_size) - 1; step++) {
                // TODO: need check why enqueue hang, when the dependent event in status CL_SUBMITTED
                sub_mem_mgr->user_events[w_rank] = clCreateUserEvent(local_context, &ret);
                if (ret != CL_SUCCESS) {
                    OPENVINO_THROW("create null event failed!");
                }

                auto src_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                auto src_buf = src_mem->get_buffer().get();

                int32_t send_chunk_idx = (w_rank - step + w_size) % w_size;
                int32_t target_chunk_idx = (w_rank - step - 1 + w_size) % w_size;

                auto dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank]
                                                          .remote_mems[0]);  // mapped from remote memory of target rank

                size_t src_off_set = 0;
                for (int32_t j = 0; j < send_chunk_idx; j++) {
                    src_off_set = src_off_set + chunk_size[j];
                }
                // view the tensor as 1D
                size_t src_region[3] = {src_off_set, 0, 0};  // start address of src orgin
                size_t dst_region[3] = {0, 0, 0};            // dst with no padding
                size_t rect[3] = {chunk_size[send_chunk_idx] / dst_height, dst_height, 1};

                ret = clEnqueueCopyBufferRect(local_queue,
                                              src_buf,
                                              dst_cl_buf,
                                              src_region,
                                              dst_region,
                                              rect,
                                              0,
                                              0,
                                              0,
                                              0,
                                              1,
                                              &sub_mem_mgr->user_events[w_rank],
                                              &sub_mem_mgr->step1_copy_events[w_rank]);
                if (ret != CL_SUCCESS) {
                    GPU_DEBUG_TRACE_DETAIL << "scatter stage clEnqueueCopyBufferRect failed: " << ", step = " << step;
                    OPENVINO_THROW("scatter stage in syn tensor failed");
                }
                sub_mem_mgr->step1_copy_events_ptr[w_rank] =
                ocl_stream.create_event(cl::Event(sub_mem_mgr->step1_copy_events[w_rank]));

                sub_mem_mgr->step1_copy_done.fetch_add(1);
                while (true) {
                    if (sub_mem_mgr->step1_copy_done.load() == 2)
                        break;
                }

                auto dst_mem_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                    sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                auto dst_cl_buf_add = dst_mem_add->get_buffer().get();
                auto& adder_instance = get_adder_instance(w_rank);

                auto src_mem_add = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[1];
                auto src_cl_buf_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem_add)->get_buffer().get();

                size_t off_set_add = 0;
                for (int32_t j = 0; j < target_chunk_idx; j++)
                    off_set_add = off_set_add + chunk_size[j];
                adder_instance.tensor_add_sub_1(
                    local_stream,
                    src_cl_buf_add,
                    dst_cl_buf_add,
                    chunk_size[target_chunk_idx] / output_element_size,
                    adder_instance.element_type_to_kernel_data_type(dst_mem_add->get_layout().data_type),
                    off_set_add / output_element_size,
                    1,
                    &sub_mem_mgr->step1_copy_events[dst_idx],
                    &sub_mem_mgr->step2_add_events[w_rank]);

                sub_mem_mgr->step2_add_events_ptr[w_rank] =
                    ocl_stream.create_event(cl::Event(sub_mem_mgr->step2_add_events[w_rank]));
                sub_mem_mgr->step2_add_done.fetch_add(1);
                while (true) {
                    if (sub_mem_mgr->step2_add_done.load() == 2)
                        break;
                }
            }
            timer.measure_end(std::string("scatter stage for Ring all-reduce"));
            instance.all_reduce_broadcast_times = timer.get().count();
            timer.measure_start(std::string("gather stage for Ring all-reduce"));
            for (int32_t step = 0; step < static_cast<int>(w_size) - 1; step++) {
                int32_t send_chunk_idx = (w_rank - step + 1) % w_size;
                int32_t recv_chunk_idx = (w_rank - step) % w_size;
                {
                    auto src_mem = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                        sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                    auto src_buf = src_mem->get_buffer().get();

                    cl_mem dst_cl_buf = static_cast<cl_mem>(sub_mem_mgr->_memorys_table[id][w_rank].remote_mems[0]);
                    size_t off_set = 0;
                    for (int32_t j = 0; j < send_chunk_idx; j++) {
                        off_set = off_set + chunk_size[j];
                    }
                    size_t src_rec[3] = {off_set, 0, 0};
                    size_t dst_rec[3] = {0, 0, 0};
                    size_t rect[3] = {chunk_size[send_chunk_idx] / dst_height, dst_height, 1};
                    ret = clEnqueueCopyBufferRect(local_queue,
                                                  src_buf,
                                                  dst_cl_buf,
                                                  src_rec,
                                                  dst_rec,
                                                  rect,
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  2,
                                                  sub_mem_mgr->step2_add_events,
                                                  &sub_mem_mgr->step3_gather_copy_events[w_rank]);
                    if (ret != CL_SUCCESS) {
                        GPU_DEBUG_TRACE_DETAIL
                            << "broadcast of gather stage clEnqueueCopyBufferRect failed: " << ", step = " << step;
                        OPENVINO_THROW("broadcast stage in sync tensor failed: ");
                    }
                    sub_mem_mgr->step3_gather_copy_events_ptr[w_rank] =
                    ocl_stream.create_event(cl::Event(sub_mem_mgr->step3_gather_copy_events[w_rank]));

                    sub_mem_mgr->step3_concat_copy_done.fetch_add(1);
                    while (true) {
                        if (sub_mem_mgr->step3_concat_copy_done.load() == 2)
                            break;
                    }
                }
                {
                    auto dst_mem_add = std::dynamic_pointer_cast<const ocl::gpu_buffer>(
                        sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[0]);
                    auto dst_cl_buf_add = dst_mem_add->get_buffer().get();

                    auto src_mem_add = sub_mem_mgr->_memorys_table[id][w_rank].recv_bufs[1];
                    auto src_cl_buf_add =
                        std::dynamic_pointer_cast<const ocl::gpu_buffer>(src_mem_add)->get_buffer().get();
                    size_t off_set = 0;
                    for (int32_t j = 0; j < recv_chunk_idx; j++) {
                        off_set = off_set + chunk_size[j];
                    }
                    size_t dst_rec[3] = {off_set, 0, 0};
                    size_t src_rec[3] = {0, 0, 0};
                    size_t rect[3] = {chunk_size[recv_chunk_idx] / dst_height, dst_height, 1};
                    auto ret = clEnqueueCopyBufferRect(local_queue,
                                                       src_cl_buf_add,
                                                       dst_cl_buf_add,
                                                       src_rec,
                                                       dst_rec,
                                                       rect,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       1,
                                                       &sub_mem_mgr->step3_gather_copy_events[dst_idx],
                                                       &sub_mem_mgr->step4_gather_copy_events[w_rank]);
                    if (ret != CL_SUCCESS) {
                        std::cout << "gather stage clEnqueueCopyBufferRect failed: " << ", step = " << step;
                        OPENVINO_THROW("gather stage of sync tensor failed: ");
                    }
                    sub_mem_mgr->step4_gather_copy_events_ptr[w_rank] =
                    ocl_stream.create_event(cl::Event(sub_mem_mgr->step4_gather_copy_events[w_rank]));

                    ret = clSetUserEventStatus(sub_mem_mgr->user_events[w_rank], CL_COMPLETE);
                    if (ret != CL_SUCCESS) {
                        OPENVINO_THROW("sync tensor trigger all-reduce execute failed!");
                    }
                    sync_events.push_back(sub_mem_mgr->step4_gather_copy_events_ptr[w_rank]);
                }
            }
            timer.measure_end(std::string("gather stage for Ring all-reduce"));
            instance.all_reduce_gather_times = timer.get().count();
        } else {
            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].flag) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }

            auto src_p2p_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.get_output_memorys()[1]);
            auto src_cl_buf = src_p2p_buf->get_buffer().get();

            auto& ocl_stream = downcast<ocl::ocl_stream>(local_stream);
            auto queue = ocl_stream.get_cl_queue().get();

            auto split_parts = [](int len, int n) {
                int average = len / n;
                std::vector<int> parts(n, average);
                parts.back() = len - average * (n - 1);
                return parts;
            };
            auto output_layout = instance.get_output_layout(0);
            ov::element::Type output_element_type = output_layout.data_type;
            auto output_element_size = output_element_type.size();
            auto output_shape = output_layout.get_shape();
            auto sub_out_dim_vec = split_parts(output_shape[-1], w_size);

            int32_t off_set = 0;
            for (int32_t j = 0; j < w_rank; j++) {
                off_set = off_set + sub_out_dim_vec[j];
            }
            size_t src_rec[3] = {0, 0, 0};
            size_t dst_rec[3] = {off_set * output_element_size, 0, 0};
            size_t rect[3] = {sub_out_dim_vec[w_rank] * output_element_size, 1, output_shape[0]};

            if (w_rank == 0) {
                std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                if (sub_mem_mgr->result != nullptr) {
                    free(sub_mem_mgr->result);
                    sub_mem_mgr->result = nullptr;
                }
                sub_mem_mgr->result = malloc(instance.get_output_memorys()[0]->size());
                sub_mem_mgr->updated_flag = true;
            } else {
                while (true) {
                    std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
                    if (sub_mem_mgr->updated_flag)
                        break;
                }
            }

            auto ret = clEnqueueReadBufferRect(queue,
                                               src_cl_buf,
                                               CL_TRUE,
                                               src_rec,
                                               dst_rec,
                                               rect,
                                               sub_out_dim_vec[w_rank] * output_element_size,
                                               sub_out_dim_vec[w_rank] * output_element_size,
                                               output_shape[-1] * output_element_size,
                                               output_shape[-1] * output_element_size,
                                               sub_mem_mgr->result,
                                               0,
                                               nullptr,
                                               nullptr);
            if (ret < 0)
                cl::detail::errHandler(ret, "EnqueueReadBufferRect");
            ////CHECK_OCL_ERROR(ret, "clEnqueueReadBufferRect failed");

            clFinish(queue);

            sub_mem_mgr->_memorys_table[id][w_rank].all_gather_flag = true;

            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].all_gather_flag == true) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }

            instance.get_output_memorys()[0]
                ->copy_from(ocl_stream, sub_mem_mgr->result, 0, 0, instance.get_output_memorys()[0]->size(), true);

            sub_mem_mgr->_memorys_table[id][w_rank].all_gather_copy_flag = true;

            while (true) {
                size_t wait_all_ouput_ready = 0;
                for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                    if (sub_mem_mgr->_memorys_table[id][idx].all_gather_copy_flag == true) {
                        wait_all_ouput_ready++;
                    }
                }
                if (wait_all_ouput_ready == w_size)
                    break;
            }
        }

        // This block MUST be put exactly at the end of this method.
        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_memorys_table[id][w_rank].layout = p2p_src_layout;
            sub_mem_mgr->_use_count[id]++;
        }
        return sync_events.size() > 0 ? local_stream.group_events(sync_events) : local_stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<sync_tensor_impl>();
    }

    std::vector<void*> all_gather_remote_dst;
    cl_mem all_gather_current_dst;
};

namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::ocl, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
