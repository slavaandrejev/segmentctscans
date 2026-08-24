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

potts_min_part_lib = static_library(
    'find_intensity_range'
  , 'potts-min-partition.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      , eigen_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

executable(
    'potts-model'
  , 'potts-model.cpp'
  , include_directories: project_inc
  , link_whole: [
        cuda_context_lib
      , field_io_lib
      , potts_min_part_lib
      ]
  , dependencies: [
        boost_dep
      , fmt_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      ]
  )
