set (APPLICATION_NAME mnist)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
	"mnist/main.h.in"
	"${PROJECT_SOURCE_DIR}/source/applications/mnist/main.h"
)

add_subdirectory(mnist)
