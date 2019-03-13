# port

This module contains the interface and implementations for different platforms.
All platform specific code should go here. It's not allowed to use non standard
headers in other modules.

This module splits into `port_api` and `port`. `port_api` is the interface, and
it should not depends on any other modules including `utils`.

If the code base goes large in the future, it should be split into core and
test to keep the footprint for production libs as small as possible.

Currently Linux, Darwin (MacOS, iOS etc.) are treated as POSIX. They will be
handled differently if needed.
