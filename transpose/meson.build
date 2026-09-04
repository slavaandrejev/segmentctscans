cuda_algorithms_lib = static_library(
    'cuda_algorithms'
  , 'convert_to_fp.cpp'
  , 'find_intensity_range.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options + { 'optimization' : '3' }
  )

executable(
    'transpose'
  , 'transpose.cpp'
  , include_directories: project_inc
  , link_whole: [
        cuda_context_lib
      , cuda_algorithms_lib
      ]
  , dependencies: [
        boost_dep
      , calc_hist_dep
      , clarabel_dep
      , fmt_dep
      , libtiff_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      ])
