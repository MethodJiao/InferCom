#include "stdafx.h"
#include "ExampleUtilDemo.h"
#include "direct.h"

wchar_t* ExampleUtilDemo::getProjectPathCW()
{
	char* path = nullptr;
	path = _getcwd(NULL, 1);
	puts(path);
	const size_t cSize = strlen(path) + 1;
	wchar_t* wPath = new wchar_t[cSize];
	size_t convertedChars = 0;
	mbstowcs_s(&convertedChars, wPath, cSize, path, _TRUNCATE);
	return wPath;
}

