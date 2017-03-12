@echo off
echo Downloading CUDA toolkit 8
appveyor DownloadFile  https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe -FileName cuda_8.0.44_windows.exe
echo Installing CUDA toolkit 8
cuda_8.0.44_windows.exe -s compiler_8.0 ^
                           cublas_8.0 ^
                           cublas_dev_8.0 ^
                           cudart_8.0 ^
                           curand_8.0 ^
                           curand_dev_8.0 ^
                           visual_studio_integration_8.0

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\cudart64_80.dll" ( 
echo "Failed to install CUDA"
exit /B 1
)
                           
echo Downloading cuDNN
appveyor DownloadFile https://www.dropbox.com/s/9t56hoewk7p0hfj/cudnn-8.0-windows7-x64-v5.1.zip?dl=1 -FileName cudnn-8.0-windows7-x64-v5.1-zip
7z x cudnn-8.0-windows7-x64-v5.1-zip -ocudnn

copy cudnn\cuda\bin\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin"
copy cudnn\cuda\lib\x64\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64"
copy cudnn\cuda\include\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"



set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp;%PATH%
set CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0
set CUDA_PATH_V8_0=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0

nvcc -V

cd ..
git clone https://github.com/Microsoft/vcpkg.git vcpkg
cd vcpkg
mkdir downloads
copy nul downloads\AlwaysAllowDownloads
powershell -exec bypass scripts\bootstrap.ps1
vcpkg integrate install
vcpkg install gtest:x64-windows

cd "%APPVEYOR_BUILD_FOLDER%"