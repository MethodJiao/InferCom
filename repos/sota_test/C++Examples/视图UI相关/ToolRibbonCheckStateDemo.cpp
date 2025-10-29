#include "stdafx.h"
#include "ToolRibbonCheckStateDemo.h"
using namespace BIMBase::FrameWork;

ToolRibbonCheckStateDemo::ToolRibbonCheckStateDemo()
{
}


ToolRibbonCheckStateDemo::~ToolRibbonCheckStateDemo()
{
}

void ToolRibbonCheckStateDemo::setCheckState()
{
	BPRibbonUtil::ribbonSetCheckCtrlState(_T("ProjectTreeDemo"), true);
	AfxMessageBox(L"“—…Ë÷√");
 }


AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("checkStateDemo"), &ToolRibbonCheckStateDemo::setCheckState);
AutoDoRegisterFunctionsEnd
