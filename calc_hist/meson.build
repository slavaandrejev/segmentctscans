# takes too long in debug mode, always opimize this
calc_hist_lib = static_library(
    'calc_hist'
  , 'calc_hist.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: {
        'optimization' : '3'
      } + cuda_overr_options
  )
