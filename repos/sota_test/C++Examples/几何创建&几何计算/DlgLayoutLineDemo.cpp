// DlgLayoutLineDemo.cpp: 实现文件
//

#include "stdafx.h"
#include "Examples.h"
#include "DlgLayoutLineDemo.h"
#include "afxdialogex.h"
#include "resource.h"


// DlgLayoutLineDemo 对话框

IMPLEMENT_DYNAMIC(DlgLayoutLineDemo, CDialogEx)

DlgLayoutLineDemo::DlgLayoutLineDemo(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_LAYOUT_LINE, pParent)
{
	m_nInputWay = 0;
}

DlgLayoutLineDemo::~DlgLayoutLineDemo()
{}

void DlgLayoutLineDemo::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_RADIO_LAYOUT, m_btLayout);
	DDX_Control(pDX, IDC_RADIO_DRAW, m_btDraw);
	DDX_Control(pDX, IDC_RADIO_CONTINUEDRAW, m_btContinueDraw);
}

BEGIN_MESSAGE_MAP(DlgLayoutLineDemo, CDialogEx)
	ON_BN_CLICKED(IDC_RADIO_LAYOUT, &DlgLayoutLineDemo::OnBnClickedLayout)
	ON_BN_CLICKED(IDC_RADIO_DRAW, &DlgLayoutLineDemo::OnBnClickedDraw)
	ON_BN_CLICKED(IDC_RADIO_CONTINUEDRAW, &DlgLayoutLineDemo::OnBnClickedRadioContinuedraw)
END_MESSAGE_MAP()

void DlgLayoutLineDemo::OnBnClickedLayout()
{
	m_btDraw.SetCheck(false);
	m_btLayout.SetCheck(true);
	m_btContinueDraw.SetCheck(false);
	m_nInputWay = 0;
}

void DlgLayoutLineDemo::OnBnClickedDraw()
{
	m_btDraw.SetCheck(true);
	m_btLayout.SetCheck(false);
	m_btContinueDraw.SetCheck(false);
	m_nInputWay = 1;
}

void DlgLayoutLineDemo::OnBnClickedRadioContinuedraw()
{
	m_btDraw.SetCheck(false);
	m_btLayout.SetCheck(false);
	m_btContinueDraw.SetCheck(true);
	m_nInputWay = 2;
}

BOOL DlgLayoutLineDemo::OnInitDialog()
{
	CRect rect = CRect(250, 200, 450, 400);
	this->MoveWindow(rect);
	CDialogEx::OnInitDialog();
	m_btLayout.SetCheck(true);
	m_btDraw.SetCheck(false);
	m_btContinueDraw.SetCheck(false);
	return true;
}



BOOL DlgLayoutLineDemo::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 在此添加专用代码和/或调用基类
	if (pMsg->wParam == VK_TAB || pMsg->wParam == VK_SHIFT)
	{
		UINT32 nIndex = BPViewManager::getInstance().getActiveIndex();
		CFrameWnd* pFrameWnd = BIMBase::FrameWork::BPUIFrameWorkUtil::getView(nIndex);
		if (!pFrameWnd)
		{
			return CDialogEx::PreTranslateMessage(pMsg);
		}
		CView* pView = pFrameWnd->GetActiveView();
		if (!pView)
		{
			return CDialogEx::PreTranslateMessage(pMsg);
		}
		::SendMessage(pView->GetSafeHwnd(), pMsg->message, pMsg->wParam, pMsg->lParam);
		return true;
	}
	return CDialogEx::PreTranslateMessage(pMsg);
}

