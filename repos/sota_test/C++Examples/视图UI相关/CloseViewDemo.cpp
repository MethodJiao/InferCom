#include "stdafx.h"
#include "CloseViewDemo.h"


void CloseViewdemo::CloseViewDemo()
{
	//获取活跃的view的number
	int nNum = BIMBase::FrameWork::BPUIFrameWorkUtil::getActiveViewIndex();

	//关闭视图
	CFrameWnd* pView = BIMBase::FrameWork::BPUIFrameWorkUtil::getView(nNum);
	if (pView != nullptr)
		pView->SendMessage(WM_CLOSE); //post message
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("closeViewDemo", &CloseViewdemo::CloseViewDemo);
AutoDoRegisterFunctionsEnd