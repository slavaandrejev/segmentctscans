project(
    'kapitza-pendulum'
  , ['c', 'cpp']
  , version : '0.1'
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
  , language: ['cpp']
  )

compiler = meson.get_compiler('cpp')
if compiler.get_id().contains('clang')
    add_project_arguments('-Wno-mathematical-notation-identifier-extension', language: 'cpp')
endif

boost_dep    = dependency('boost', modules: ['atomic', 'filesystem', 'headers', 'iostreams', 'program_options', 'random', 'regex'])
clarabel_dep = dependency('Clarabel', method: 'cmake', modules: ['Clarabel::shared'])
fmt_dep      = dependency('fmt')
libtiff_dep  = dependency('libtiff-4')
mdspan_dep   = dependency('mdspan', method: 'cmake')
omp_dep      = dependency('openmp')
opencv_dep   = dependency('OpenCV', method: 'cmake')
range_dep    = dependency('range-v3', method: 'cmake')

subdir('totalvarmin2d')
subdir('totalvarmin')