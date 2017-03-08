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

We are eager to see you pull requests, so that together we can build a simple and powerful
deep learning framework!

## Building from source

| Branch | Status |
|--------|:-------|
| master | [![Build status](https://ci.appveyor.com/api/projects/status/8ohrw1c7leli7lmr/branch/master?svg=true)](https://ci.appveyor.com/project/olegtarasov/retia/branch/master) |
| develop | [![Build status](https://ci.appveyor.com/api/projects/status/8ohrw1c7leli7lmr/branch/develop?svg=true)](https://ci.appveyor.com/project/olegtarasov/retia/branch/develop) |

Retia is following the F5 build paradigm: you should just open the solution in Visual Studio,
hit F5 and it should run. You can build CPU-only version just like that. All you need is a C++
compiler installed as part of Visual Studio.

However, if you are building with GPU support, there are some heavy-duty
dependencies which you should install by yourself. Refer to Wiki page [Building from source](https://github.com/total-world-domination/Retia/wiki/Building-from-source)
for detailed build instructions.

### Project dependencies and libraries

We also use some neat tools and libraries, so you might want to read this 
[wiki page](https://github.com/total-world-domination/Retia/wiki/Project-dependencies-and-libraries).

## Examples

Of course you want some examples! The great place to start is a 
[character-level language model](https://github.com/total-world-domination/Retia/wiki/Language-model-example).
If fact, this is the only example so far, but there will be plenty more! :)

## Interface compatibility policy

Since Retia is in an early stage of development, please expect breaking changes in public interfaces.
We feel that right now it's better to focus on development speed rather than strict backward compatibility.

Of cource when the library reaches its first stable version, we will develop and uphold rather strict
compatibility policy.