To build Retia.Cuda.Tests, you need to install vcpkg and Google Test. To do this you will need:

* Visual Studio 2015 Update 3 or higher
* git

Now open admin command prompt and execute:

```
git clone https://github.com/Microsoft/vcpkg.git [vcpkg install dir]
cd [vcpkg install dir]
powershell -exec bypass scripts\bootstrap.ps1
vcpkg integrate install
vcpkg install gtest gtest:x64-windows
```

Restart Visual Studio if you had it open. Now you can hopefully build the test project.