# always compile IO with optimization, otherwise it is too slow
field_io_lib = static_library(
    'field_io'
  , 'field-io.cpp'
  , include_directories: project_inc
  , build_by_default: false
  , override_options: {
        'optimization' : '3'
      }
  )

cuda_algorithms_lib = static_library(
    'cuda_algorithms'
  , 'chambolle.cpp'
  , include_directories: project_inc
  , dependencies: [
        boost_dep
      , cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

executable(
    'totalvarmin'
  , 'totalvarmin.cpp'
  , include_directories: project_inc
  , link_whole: [
        field_io_lib
      , cuda_algorithms_lib
      , cuda_context_lib
      ]
  , dependencies: [
        boost_dep
      , calc_hist_dep
      , fmt_dep
      , libtiff_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      ])
