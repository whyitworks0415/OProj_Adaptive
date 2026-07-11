// stb_image 구현체 정의 – 이 .cpp 파일 하나에서만 STB_IMAGE_IMPLEMENTATION 을 정의한다.
// GLTFLoader.cpp 가 tinygltf 를 통해 stbi_* 함수를 사용하므로 링크 단계에 필요하다.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
