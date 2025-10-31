#include "stdafx.h"
#include "DlgArchWallDemo.h"
#include "ToolArchWallDemo.h"

static DlgArchWallDemo* m_dlg;
ToolArchWallDemo::ToolArchWallDemo()
{
	m_eLayoutWay = ToolLayoutCubeDemo::CubeLayoutWay::OnePoint;
	m_nHeight = 3000;
	m_nLength = 1000;
	m_nWidth = 200;
}


ToolArchWallDemo::~ToolArchWallDemo()
{
	if (m_dlg != nullptr)
		m_dlg->ShowWindow(SW_HIDE);
}

void ToolArchWallDemo::_onPostInstall()
{
	//调用基类
	T_Super::_onPostInstall();

	PBBimModuleResourceOverride resOverride;
	if (m_dlg == nullptr)
	{
		CView* pView = ((CFrameWnd*)(AfxGetApp()->m_pMainWnd))->GetActiveFrame()->GetActiveView();
		m_dlg = new DlgArchWallDemo;
		m_dlg->Create(DlgArchWallDemo::IDD, pView);
		m_dlg->ShowWindow(SW_SHOW);
		m_eLayoutWay = (ToolLayoutCubeDemo::CubeLayoutWay)m_dlg->m_nInputWay;
	}
	else
		m_dlg->ShowWindow(SW_SHOW);
	//打开捕捉
	
