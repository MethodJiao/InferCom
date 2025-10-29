#include "stdafx.h"
#include "ToolExportFBXDemo.h"
#include "DataExchange/FbxUtilTool.h"
#pragma comment(lib, "BPDataExchange.lib")

using namespace DataExchange;

void ToolExportFBXDemo::exportFbx()
{
	//保存路径
	std::wstring sPath = _T("D:\\model");
	//导出当前工程为fbx文件
	FbxUtilTool::exportAll2Fbx(sPath);
}

//注册命令
AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("exportFbxDemo", ToolExportFBXDemo::exportFbx);
AutoDoRegisterFunctionsEnd