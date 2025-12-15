// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/nn/parallel/pp/send_recv.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

float Schedule1F1B::StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                     const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                     const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    const int n = num_micro_batches_;
    if (n == 0) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    const int num_stages = stage_->num_stages();
    const int stage_index = stage_->stage_index();

    const int warmup_steps = num_stages;
    const int cooldown_steps = num_stages;
    const int total_steps = num_stages + n - 1;

    std::vector<std::vector<std::shared_ptr<Tensor>>> activations(n);

    int mb_forward_i;  // forward micro_batch index
    int mb_backward_i; // backward micro_batch index
    printf("[stage %d] warmup_steps start\n", stage_index);
    // warmup_steps
    for (mb_forward_i = 0, mb_backward_i = 0; mb_forward_i < std::min(n, warmup_steps);
         ++mb_forward_i, ++mb_backward_i) {
        std::vector<std::shared_ptr<Tensor>> inputs;
        if (stage_->IsFirstStage()) {
            inputs = {microbatch_inputs[mb_forward_i]};
        } else {
            inputs = ReceiveFromPrev(stage_->prev_rank());
        }

        activations[mb_forward_i] = stage_->ForwardOneChunk(inputs);

        if (!stage_->IsLastStage()) {
            SendToNext(activations[mb_forward_i], stage_->next_rank());
        } else {
            auto target = microbatch_targets[mb_backward_i];
            auto output = activations[mb_backward_i][0];
            auto target_on_device = target->To(output->GetDevice());
            auto loss = loss_fn->Forward({output, std::make_shared<Tensor>(target_on_device)})[0];
            loss = loss / n;

            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            total_loss += static_cast<const float *>(loss_cpu.DataPtr())[0];

            printf("warmup_steps start Backward\n");
            loss->Backward();
        }
    }

    if (!stage_->IsLastStage()) {
        for (mb_backward_i = 0; mb_backward_i <= stage_index && mb_backward_i < n; ++mb_backward_i) {
            auto out_tensor = activations[mb_backward_i][0];

            auto gradient = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());

            out_tensor->Backward(gradient);
        }
    }

    printf("[stage %d] steady_steps start\n", stage_index);
    // steady_steps
    for (; mb_forward_i < n; ++mb_forward_i, ++mb_backward_i) {
        // Forward
        // printf("[stage %d] steady_steps mb_forward_i %d\n", stage_index_, mb_forward_i);
        std::vector<std::shared_ptr<Tensor>> inputs;
        if (stage_->IsFirstStage()) {
            inputs = {microbatch_inputs[mb_forward_i]};
        } else {
            inputs = ReceiveFromPrev(stage_->prev_rank());
        }

        activations[mb_forward_i] = stage_->ForwardOneChunk(inputs);

        printf("[stage %d] steady_steps 开始反向 mb_forward_i: %d mb_backward_i: %d\n", stage_index, mb_forward_i,
               mb_backward_i);
        // Backward
        if (!stage_->IsLastStage()) {
            SendToNext(activations[mb_forward_i], stage_->next_rank());

            auto out_tensor = activations[mb_backward_i][0];

            auto gradient = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());

            out_tensor->Backward(gradient);
        } else {
            auto target = microbatch_targets[mb_backward_i];
            auto output = activations[mb_backward_i][0];
            auto target_on_device = target->To(output->GetDevice());
            auto loss = loss_fn->Forward({output, std::make_shared<Tensor>(target_on_device)})[0];
            loss = loss / n;

            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            total_loss += static_cast<const float *>(loss_cpu.DataPtr())[0];

            loss->Backward();
        }
    }

    printf("[stage %d] cooldown_steps start\n", stage_index);
    // cooldown_steps
    if (!stage_->IsLastStage()) {
        for (; mb_backward_i < n; ++mb_backward_i) {
            auto out_tensor = activations[mb_backward_i][0];

            auto gradient = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());

            out_tensor->Backward(gradient);
        }
    }

    return total_loss;
}

void PrintScheduleTable(int n, int num_stages, int vpp_size) {
    int total_global_chunks = num_stages * vpp_size;
    int total_steps = n + total_global_chunks - 1;

    // auto schedule = PipelineParallelScheduler::GenerateGPipeSchedule(n, num_stages, vpp_size);
    auto schedule = PipelineParallelScheduler::GenerateInterleaved1F1BSchedule(n, num_stages, vpp_size);

    printf("=== 1F1B Interleaved Schedule Table ===\n");
    printf("n=%d, stages=%d, vpp=%d, total_chunks=%d \n\n", n, num_stages, vpp_size, total_global_chunks);

    printf("Step | Type  | Microbatch | Global Chunk | Local Chunk | Stage\n");
    printf("-----|-------|------------|--------------|-------------|-------\n");

    for (const auto &task : schedule) {
        int owning_stage = task.global_chunk_id % num_stages;
        int local_chunk = task.global_chunk_id / num_stages;

        printf("%4d | %-6s| %-11d| %-13d| %-12d| %d\n", task.step, task.is_forward ? "Forward" : "Backward",
               task.microbatch_id, task.global_chunk_id, local_chunk, owning_stage);
    }
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::ReceiveFromPrev(int peer_rank) {
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    auto &shapes = stage_->recv_shape();
    for (size_t i = 0; i < shapes.size(); ++i) {
        // FIXME(jym): The data type between stages is not float32, which will cause a crash
        auto tensor = std::make_shared<Tensor>(shapes[i], DataType::kFLOAT32, stage_->device());
        tensor->set_requires_grad(true);
        recv_tensors.push_back(tensor);
    }

    return IRecv(recv_tensors, stage_->device(), stage_->stage_index(), peer_rank);
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                                  int peer_rank) {
    return ISend(tensors, stage_->device(), stage_->stage_index(), peer_rank, stage_->recv_shape());
}

PipelineParallelScheduler::Task PipelineParallelScheduler::CreateTask(int step, int mb, int global_chunk,
                                                                      int num_stages, int total_chunks,
                                                                      bool is_forward) {
    PipelineParallelScheduler::Task task;
    task.step = step;
    task.microbatch_id = mb;
    task.global_chunk_id = global_chunk;
    task.local_chunk_idx = global_chunk / num_stages;
    task.is_forward = is_forward;
    task.stage_id = global_chunk % num_stages;
    task.is_last_chunk = (global_chunk == total_chunks - 1);
    task.is_first_chunk = (global_chunk == 0);
    return task;
}

std::vector<PipelineParallelScheduler::Task> PipelineParallelScheduler::GenerateGPipeSchedule(int n, int num_stages,
                                                                                              int vpp_size) {
    std::vector<Task> schedule;
    int total_global_chunks = num_stages * vpp_size;
    int total_steps = n + total_global_chunks - 1;

    // ======== Forward Pass ========
    for (int step = 0; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int global_chunk_id = step - mb;
            if (global_chunk_id >= 0 && global_chunk_id < total_global_chunks) {
                auto is_forward = true;
                auto task = CreateTask(step, mb, global_chunk_id, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }
        }
    }

    // ======== Backward Pass ========
    for (int step = 0; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int global_chunk_id = (total_steps - 1 - step) - mb;
            if (global_chunk_id >= 0 && global_chunk_id < total_global_chunks) {
                auto is_forward = false;
                auto task
                    = CreateTask(step + total_steps, mb, global_chunk_id, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }
        }
    }

    // sorted according to step、local_chunk_idx
    std::sort(schedule.begin(), schedule.end(), [](const Task &a, const Task &b) {
        if (a.step != b.step) {
            return a.step < b.step;
        }

        return a.local_chunk_idx < b.local_chunk_idx;
    });

    return schedule;
}

std::vector<PipelineParallelScheduler::Task>
PipelineParallelScheduler::GenerateInterleaved1F1BSchedule(int n, int num_stages, int vpp_size) {
    std::vector<Task> schedule;

    if (n <= 0 || num_stages <= 0 || vpp_size <= 0) {
        return schedule;
    }

    int total_global_chunks = num_stages * vpp_size;

    int warmup_steps = total_global_chunks - 1;
    int total_steps = 2 * warmup_steps + n;

    // std::cout << "Interleaved 1F1B Parameters:" << std::endl;
    // std::cout << "  n = " << n << ", num_stages = " << num_stages << ", vpp_size = " << vpp_size << std::endl;
    // std::cout << "  total_virtual_stages = " << total_global_chunks << std::endl;
    // std::cout << "  warmup_steps = " << warmup_steps << std::endl;
    // std::cout << "  total_steps = " << total_steps << std::endl;

    // ================ Warm-up ================
    for (int step = 0; step < warmup_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int forward_global_chunk = step - mb;
            if (forward_global_chunk >= 0 && forward_global_chunk < total_global_chunks) {
                auto is_forward = true;
                auto task = CreateTask(step, mb, forward_global_chunk, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }
        }
    }

    // ================ Steady ================
    for (int step = warmup_steps; step < warmup_steps + n; ++step) {
        int stable_step = step - warmup_steps; // 稳定阶段内的步数

        for (int mb = 0; mb < n; ++mb) {
            // Forward
            int forward_global_chunk = step - mb;
            if (forward_global_chunk >= 0 && forward_global_chunk < total_global_chunks) {
                auto is_forward = true;
                auto task = CreateTask(step, mb, forward_global_chunk, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }

            // Backward
            int backward_global_chunk = (total_global_chunks - 1) - (stable_step - mb);

            if (backward_global_chunk >= 0 && backward_global_chunk < total_global_chunks) {
                auto is_forward = false;
                auto task = CreateTask(step, mb, backward_global_chunk, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }
        }
    }

    // ================ Cool-down ================
    for (int step = warmup_steps + n; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int backward_step = step - (warmup_steps);
            int backward_global_chunk = (total_global_chunks - 1) - (backward_step - mb);
            if (backward_global_chunk >= 0 && backward_global_chunk < total_global_chunks) {
                auto is_forward = false;
                auto task = CreateTask(step, mb, backward_global_chunk, num_stages, total_global_chunks, is_forward);
                schedule.push_back(task);
            }
        }
    }
    // std::cout << "Cool-down阶段 OK" << std::endl;

    return schedule;
}

float PipelineSchedule::StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                         const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                         const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    int n = num_micro_batches_;
    int num_stages = stage_->num_stages();
    int stage_idx = stage_->stage_index();
    int vpp_size = global::GetVirtualPipelineParallelSize();

    auto schedule = PipelineParallelScheduler::GenerateInterleaved1F1BSchedule(n, num_stages, vpp_size);

    // if (stage_idx == 0) {
    //     PrintScheduleTable(n, num_stages, vpp_size);
    // }
    float total_loss = 0.0f;
    // printf("[stage %d] Schedule has %lu tasks\n", stage_idx, schedule.size());

    std::vector<std::vector<std::vector<std::shared_ptr<Tensor>>>> activations(
        vpp_size, std::vector<std::vector<std::shared_ptr<Tensor>>>(n));

    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto &task = schedule[i];
        if (task.stage_id != stage_idx) {
            continue;
        }

        int mb = task.microbatch_id;
        if (task.is_forward) {
            infini_train::AutocastGuard autocast_guard(stage_->device()->Type(), dtype);

            std::vector<std::shared_ptr<Tensor>> inputs;

            if (task.is_first_chunk) {
                inputs = {microbatch_inputs[mb]};
            } else {
                if (stage_->IsFirstStage()) {
                    inputs = ReceiveFromPrev(num_stages - 1);
                } else {
                    inputs = ReceiveFromPrev(stage_->prev_rank());
                }
            }

            activations[task.local_chunk_idx][mb] = stage_->ForwardOneChunk(inputs, task.local_chunk_idx);

            if (!task.is_last_chunk) {
                if (stage_->IsLastStage()) {
                    SendToNext(activations[task.local_chunk_idx][mb], 0);
                } else {
                    SendToNext(activations[task.local_chunk_idx][mb], stage_->next_rank());
                }
            }
        } else {
            if (task.is_last_chunk) {
                auto target = microbatch_targets[mb];
                std::shared_ptr<Tensor> loss;
                {
                    infini_train::AutocastGuard autocast_guard(stage_->device()->Type(), dtype);

                    auto target_on_device = target->To(activations[task.local_chunk_idx][mb][0]->GetDevice());
                    loss = loss_fn->Forward(
                        {activations[task.local_chunk_idx][mb][0], std::make_shared<Tensor>(target_on_device)})[0];
                    loss = loss / n;
                }
                total_loss
                    += static_cast<const float *>(loss->To(DeviceManager::Instance()->GetDefaultDevice()).DataPtr())[0];

                // printf("[stage %d][chunk %d] ------------最后一个chunk反向 microbatch %d-----------------------  \n",
                //        stage_idx, task.local_chunk_idx, mb);
                loss->Backward();
            } else {
                auto out_tensor = activations[task.local_chunk_idx][mb][0];

                auto dummy_gradient
                    = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());

                out_tensor->Backward(dummy_gradient);
            }
        }
    }

    return total_loss;
}

float PipelineSchedule::Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                             const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    std::vector<std::shared_ptr<Tensor>> micro_batches(num_micro_batches_);
    std::vector<std::shared_ptr<Tensor>> target_mbs(num_micro_batches_);
    if (stage_->IsFirstStage()) {
        micro_batches = input->Split(input->Dims()[0] / num_micro_batches_);
    }

    if (stage_->IsLastStage()) {
        target_mbs = target->Split(target->Dims()[0] / num_micro_batches_);
    }

    const auto &optimizer = stage_->optimizer();

    optimizer->ZeroGrad();

    float lossf = StepMicroBatches(micro_batches, target_mbs, loss_fn, dtype);

    optimizer->Step();

    return lossf;
}

} // namespace infini_train::nn::parallel
