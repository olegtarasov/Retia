@echo off
echo Downloading CUDA toolkit 9
appveyor DownloadFile  https://www.dropbox.com/s/qfgvrf9i109mdsh/cuda9.zip?dl=1
7z x cuda9.zip -ocuda
cd cuda
echo Installing CUDA toolkit 9
setup.exe -s compiler_9.0 ^
                           cublas_9.0 ^
                           cublas_dev_9.0 ^
                           cudart_9.0 ^
                           curand_9.0 ^
                           curand_dev_9.0 ^
                           visual_studio_integration_9.0

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\cudart64_90.dll" ( 
echo "Failed to install CUDA"
exit /B 1
)

rem echo Installing VS integration
rem copy _vs\*.* "c:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V140\BuildCustomizations"

rem cd ..
                           
echo Downloading cuDNN
appveyor DownloadFile https://www.dropbox.com/s/984rz70pnjnclk2/cudnn7.zip?dl=1
7z x cudnn7.zip -ocudnn

copy cudnn\cuda\bin\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin"
copy cudnn\cuda\lib\x64\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64"
copy cudnn\cuda\include\*.* "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include"

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
set CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0
set CUDA_PATH_V9_0=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v9.0

nvcc -V

cd ..
git clone https://github.com/Microsoft/vcpkg.git vcpkg
cd vcpkg
mkdir downloads
copy nul downloads\AlwaysAllowDownloads
powershell.exe -NoProfile -ExecutionPolicy Bypass "& {& 'scripts\bootstrap.ps1'}"
vcpkg integrate install
vcpkg install gtest:x64-windows

cd "%APPVEYOR_BUILD_FOLDER%"