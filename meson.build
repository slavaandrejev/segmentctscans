project(
    'soil-ct-scan'
  , ['c', 'cpp']
  , version : '0.1'
  , meson_version: '>=1.2'
  , default_options: {
        'cpp_std': 'gnu++26'
      , 'c_std':   'gnu18'
      }
  )

if get_option('buildtype') == ('debug')
    add_project_arguments('-DDEBUG', language: ['c', 'cpp'])
endif
if get_option('buildtype') == 'release' or get_option('buildtype') == 'debugoptimized'
    add_project_arguments('-DNDEBUG', language: ['c', 'cpp'])
endif
add_project_arguments(
    '-DMDSPAN_IMPL_STANDARD_NAMESPACE=std'
  , '-DMDSPAN_USE_BRACKET_OPERATOR=1'
  , '-Wno-deprecated-declarations'
  , language: ['cpp']
  )

compiler = meson.get_compiler('cpp')
if compiler.get_id().contains('clang')
    add_project_arguments('-Wno-mathematical-notation-identifier-extension', language: 'cpp')
endif

project_inc = include_directories('include')

boost_dep    = dependency('boost', modules: ['archive', 'serialization', 'program_options', 'iostreams'])
cccl_dep     = dependency('cccl', method: 'cmake', modules: ['CCCL::Thrust', 'CCCL::CUB', 'CCCL::libcudacxx'])
clarabel_dep = dependency('Clarabel', method: 'cmake', modules: ['Clarabel::shared'])
cuda_dep     = dependency('cuda')
eigen_dep    = dependency('Eigen3', method: 'cmake', modules: 'Eigen3::Eigen')
fmt_dep      = dependency('fmt')
libtiff_dep  = dependency('libtiff-4')
mdspan_dep   = dependency('mdspan', method: 'cmake')
omp_dep      = dependency('openmp')
opencv_dep   = dependency('OpenCV', method: 'cmake')
range_dep    = dependency('range-v3', method: 'cmake')

cuda_cpp_args = [
    '-x', 'cuda'
  , '--cuda-gpu-arch=sm_86'
  , '-Wno-unknown-cuda-version'
  , '-Wno-unsupported-target-opt'
  ]
cuda_overr_options = {'cpp_std': 'gnu++23'}

subdir('cuda-context')
subdir('min_max')
subdir('calc_hist')
subdir('totalvarmin')
subdir('transpose')
subdir('gmm-peaks')
subdir('potts-model')
