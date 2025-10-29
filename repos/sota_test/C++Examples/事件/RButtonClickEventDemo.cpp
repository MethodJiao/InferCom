#include "stdafx.h"



void RButtonClickEventDemo::_setMenu(BIMBase::FrameWork::RButtonClickItemPtr& initMenu)
{
	//添加停靠对话框checkItem项，用于显隐停靠对话框
	bool bIsDockShow = false;
	DependencyInversion& depinv = DependencyInversion::instance("DemoGetShowOrHideDockDlg");
	if (depinv.is<bool(void)>())
	{
		auto fun = depinv.get<bool(void)>();
		bIsDockShow = fun();
	}
	initMenu->appendSubItem(L"停靠对话框", L"DemoShowOrHideDockDlg", true, bIsDockShow);

	//添加带子菜单的菜单项，所添加的子菜单与菜单中原始功能一致，只用于演示如何添加子菜单
	BIMBase::FrameWork::RButtonClickItemPtr ptrInsertItem = BIMBase::FrameWork::RButtonClickItem::create(L"二次开发CPP", L"");
	ptrInsertItem->appendSubItem(L"复制", L"PBBIM.Tool.Copy");
	ptrInsertItem->appendSubItem(L"删除", L"PBBIM.Tool.Remove", false);
	ptrInsertItem->appendSubItem(L"移动", L"PBBIM.Tool.Move");
	initMenu->insterSubItem(L"停靠对话框", ptrInsertItem);
}