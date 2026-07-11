@echo off
REM GLSL 셰이더를 SPIR-V 바이너리로 컴파일한다.
REM Vulkan SDK의 glslc가 PATH에 있어야 한다.
REM 빌드 전에 프로젝트 루트에서 실행한다.

if not exist "shaders\spv" mkdir "shaders\spv"

echo Compiling scene.vert ...
glslc shaders/scene.vert -o shaders/spv/scene.vert.spv
if errorlevel 1 ( echo FAILED: scene.vert & exit /b 1 )

echo Compiling scene.frag ...
glslc shaders/scene.frag -o shaders/spv/scene.frag.spv
if errorlevel 1 ( echo FAILED: scene.frag & exit /b 1 )

echo Compiling scene_instanced.vert ...
glslc shaders/scene_instanced.vert -o shaders/spv/scene_instanced.vert.spv
if errorlevel 1 ( echo FAILED: scene_instanced.vert & exit /b 1 )

echo Compiling shadow.vert ...
glslc shaders/shadow.vert -o shaders/spv/shadow.vert.spv
if errorlevel 1 ( echo FAILED: shadow.vert & exit /b 1 )

echo Compiling gizmo.vert ...
glslc shaders/gizmo.vert -o shaders/spv/gizmo.vert.spv
if errorlevel 1 ( echo FAILED: gizmo.vert & exit /b 1 )

echo Compiling gizmo.frag ...
glslc shaders/gizmo.frag -o shaders/spv/gizmo.frag.spv
if errorlevel 1 ( echo FAILED: gizmo.frag & exit /b 1 )

echo Compiling gbuffer.frag ...
glslc shaders/gbuffer.frag -o shaders/spv/gbuffer.frag.spv
if errorlevel 1 ( echo FAILED: gbuffer.frag & exit /b 1 )

echo Compiling deferred_light.vert ...
glslc shaders/deferred_light.vert -o shaders/spv/deferred_light.vert.spv
if errorlevel 1 ( echo FAILED: deferred_light.vert & exit /b 1 )

echo Compiling deferred_light.frag ...
glslc shaders/deferred_light.frag -o shaders/spv/deferred_light.frag.spv
if errorlevel 1 ( echo FAILED: deferred_light.frag & exit /b 1 )

echo Compiling sky.frag ...
glslc shaders/sky.frag -o shaders/spv/sky.frag.spv
if errorlevel 1 ( echo FAILED: sky.frag & exit /b 1 )

echo Compiling post/bloom shaders ...
glslc shaders/post.vert            -o shaders/spv/post.vert.spv
if errorlevel 1 ( echo FAILED: post.vert & exit /b 1 )
glslc shaders/bloom_bright.frag    -o shaders/spv/bloom_bright.frag.spv
if errorlevel 1 ( echo FAILED: bloom_bright.frag & exit /b 1 )
glslc shaders/bloom_blur.frag      -o shaders/spv/bloom_blur.frag.spv
if errorlevel 1 ( echo FAILED: bloom_blur.frag & exit /b 1 )
glslc shaders/bloom_composite.frag -o shaders/spv/bloom_composite.frag.spv
if errorlevel 1 ( echo FAILED: bloom_composite.frag & exit /b 1 )

echo All shaders compiled successfully.
