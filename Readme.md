# Retia framework

Retia is a deep learning framework written in C# and designed to be simple and easy 
to tinker with. Unlike many other frameworks which look like black or gray boxes, you
can actually get your hands dirty with Retia. Everything is a subject for debugging,
and there are very few abstractions to dig through. Many things are represented as
plain old matrices, without all that super generic tensor stuff.

At the same time Retia is damn fast. It's fast on CPU when you just want to poke around
with your data, and it's fast on GPU leveraging the latest technologies like cuDnn.

## Features

Retia is in an early stage of development. We focused on recurrent networks in order to 
start somewhere. So now you can build deep GRU models to create language models and other 
fun stuff. We will be adding more layer types in the future. As a matter of fact, Retia is
so easily customizable that you can start making your own layers in no time at all!

We are eager to see you pull requests, so that together we can build a simple but powerful
deep learning framework!

## Building from source

You can build Retia with or without GPU support. If you just want to take a look at the source
and tinker around for a bit, build without GPU support. You can always build the full
version later.

### Building without GPU support

You will need Visual Studio with C# and basic C++ capabilities. 

The process is very simple: just build Retia.CPUOnly.sln and you are done :)

### Building with GPU support

You will need a couple of things to build Retia from source:

* Visual Studio 2015
* CUDA Toolkit from NVidia. Get it [here](https://developer.nvidia.com/cuda-downloads).
* CuDNN library. [Download](https://developer.nvidia.com/cudnn) (you will need to register a free account with NVidia).
* Visual C++ and C++\CLI (comes with Visual Studio, just be sure to check appropriate boxes during installation).

### Building CUDA tests

To build Retia.Cuda.Tests, you need to install vcpkg and Google Test. To do this you will need:

* Visual Studio 2015 Update 3 or higher
* git accesible from your command line

Now open admin command prompt and execute:

```
git clone https://github.com/Microsoft/vcpkg.git [vcpkg install dir]
cd [vcpkg install dir]
powershell -exec bypass scripts\bootstrap.ps1
vcpkg integrate install
vcpkg install gtest gtest:x64-windows
```

Restart Visual Studio if you had it open. Now you can build the test project.

### Ammy UI

Retia uses [Ammy UI](http://www.ammyui.com/) for its GUI features. You don't need to install additional tools to build
the GUI, since Ammy is included as a self-contained NuGet package. But if you want to tinker with GUI code, we highly
recommend installing Ammy extension from [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ionoy.Ammy).

### OxyPlot

Retia also uses OxyPlot for plots. It's included as Git submodule. The reason we don't use the official NuGet package is that sometimes
we contribute performance optimizations to OxyPlot and we don't want to wait until they make it to the release version :)

Our fork is in no way a custom OxyPlot version — we always keep it in sync with upstream and there are no changes that don't eventually
get accepted into upstream repository. If we encounter some change that doesn't get accepted right away, we will do everything to resolve
the issue or rollback the change.