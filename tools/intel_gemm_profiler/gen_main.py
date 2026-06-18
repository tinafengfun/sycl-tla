import sys

manifest = sys.argv[1]
output = sys.argv[2]
kernels = []
with open(manifest) as f:
    for l in f:
        if l.strip():
            kernels.append(l.strip())

runs = '\n'.join(f'  RUN({k})' for k in kernels)
main = f'''#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include <iomanip>
#include <fstream>
#include <iostream>
#include "benchmark_runner.hpp"
#if defined(SYCL_INTEL_TARGET)
#if defined(CUTLASS_BENCHMARK_USE_FILTERED_HEADER)
#include "benchmarks_sycl.filtered.hpp"
#else
#include "benchmarks_sycl.hpp"
#endif
#endif

template <class Result>
int print_direct_result(std::string const& kernel, Result const& result) {{
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT kernel=" << kernel
            << " median_tflops=" << result.tflops
            << " avg_runtime_ms=" << result.avg_runtime_ms
            << " total_runtime_ms=" << result.total_runtime_ms
            << " input_mode=" << result.input_mode
            << " workspace_bytes=" << result.workspace_bytes
            << " input_bytes_per_buffer=" << result.input_bytes_per_buffer
            << " input_pool_target_bytes=" << result.input_pool_target_bytes
            << " input_pool_buffers=" << result.input_pool_buffers
            << " fixed_vram_input=" << result.fixed_vram_input
            << " prebuilt_variants=" << result.prebuilt_variants
            << " workspace_reuse_enabled=" << result.workspace_reuse_enabled
            << " measure_iters=" << result.measure_iters
            << " warmup_iters=" << result.warmup_iters
            << " STATUS=" << (result.success ? "OK" : "FAIL")
            << std::endl;
  return result.success ? 0 : 2;
}}

std::string trim_copy(std::string value) {{
  auto first = value.find_first_not_of(" \\t\\r\\n");
  if (first == std::string::npos) {{
    return std::string();
  }}
  auto last = value.find_last_not_of(" \\t\\r\\n");
  return value.substr(first, last - first + 1);
}}

int main(int argc, const char** argv) {{
  cutlass::CommandLine cmd(argc, argv);
  std::string config_file;
  cmd.get_cmd_line_argument("config_file", config_file, std::string(""));
  if (!config_file.empty()) {{
    BenchmarkOptions options;
    options.parse(argc, argv);
    if (options.error) return -1;
    std::ifstream file(options.config_file);
    if (!file.is_open()) {{ std::cerr << "Cannot open config" << std::endl; return 1; }}
    register_gemm_benchmarks();
    std::string line;
    while (std::getline(file, line))
      if (!line.empty() && line[0] != '#') register_benchmarks<cutlass::benchmark::GEMMOptions>(line);
    file.close();
    ::benchmark::Initialize(nullptr, nullptr);
    ::benchmark::SetDefaultTimeUnit(::benchmark::kMillisecond);
    ::benchmark::RunSpecifiedBenchmarks();
    compat::wait();
    ::benchmark::Shutdown();
    return 0;
  }}

  std::string kernel; cmd.get_cmd_line_argument("kernel", kernel, std::string(""));
  std::string kernel_file; cmd.get_cmd_line_argument("kernel_file", kernel_file, std::string(""));
  if (kernel.empty() && kernel_file.empty()) {{ std::cerr << "--kernel=NAME or --kernel_file=PATH [--m=8192 --n=4096 --k=1536]" << std::endl; return 1; }}
  register_gemm_benchmarks();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  cutlass::benchmark::GEMMOptions opts;
  cmd.get_cmd_line_argument("m", opts.m, 8192); cmd.get_cmd_line_argument("n", opts.n, 4096);
  cmd.get_cmd_line_argument("k", opts.k, 1536); cmd.get_cmd_line_argument("l", opts.l, 1);
  cmd.get_cmd_line_argument("alpha", opts.alpha, 1.0f); cmd.get_cmd_line_argument("beta", opts.beta, 0.0f);
  opts.verify_library = 0; opts.split_k_slices = 0;
#define RUN(K) if (selected_kernel == #K) {{ auto result = cutlass::benchmark::BenchmarkRunnerGemm<K>().run_direct_result(opts, hw); return print_direct_result(selected_kernel, result); }}
  auto run_selected_kernel = [&](std::string const& selected_kernel) -> int {{
{runs}
    std::cerr << "NOT_FOUND kernel=" << selected_kernel << std::endl;
    return 1;
  }};
#undef RUN
  if (!kernel_file.empty()) {{
    std::ifstream file(kernel_file);
    if (!file.is_open()) {{
      std::cerr << "Cannot open kernel file: " << kernel_file << std::endl;
      return 1;
    }}
    int batch_rc = 0;
    std::string line;
    while (std::getline(file, line)) {{
      std::string selected_kernel = trim_copy(line);
      if (selected_kernel.empty() || selected_kernel[0] == '#') {{
        continue;
      }}
      int rc = run_selected_kernel(selected_kernel);
      if (rc != 0 && batch_rc == 0) {{
        batch_rc = rc;
      }}
    }}
    return batch_rc;
  }}
  return run_selected_kernel(kernel);
}}
'''
with open(output, 'w') as f:
    f.write(main)
