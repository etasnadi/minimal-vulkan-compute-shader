#!/bin/bash

mkdir -p build/shaders
glslangValidator -V simplest.comp -o build/shaders/simplest.spv
