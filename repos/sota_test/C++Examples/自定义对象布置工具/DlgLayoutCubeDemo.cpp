// DlgLayoutCubeDemo.cpp : 实现文件
//

#include "stdafx.h"
#include "DlgLayoutCubeDemo.h"
#include "afxdialogex.h"


// DlgLayoutCubeDemo 对话框

IMPLEMENT_DYNAMIC(DlgLayoutCubeDemo, CDialogEx)

DlgLayoutCubeDemo::DlgLayoutCubeDemo(CWnd* pParent /*=NULL*/)
	: CDialogEx(DlgLayoutCubeDemo::IDD, pParent)
{
	m_nInputWay = 0;
}

DlgLayoutCubeDemo::~DlgLayoutCubeDemo()
{
}

void DlgLayoutCubeDemo::updateUI()
{
	if (m_nInputWay == 0)
	{
		((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(TRUE);
		((CButton*)GetDlgItem(IDC_RADIO_TWO_PT))->SetCheck(false);
	}
	else
	{
		((CButton*)GetDlgItem(IDC_RADIO_TWO_PT))->SetCheck(TRUE);
		((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(false);
	}
}

void DlgLayoutCubeDemo::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(DlgLayoutCubeDemo, CDialogEx)
	ON_BN_CLICKED(IDC_RADIO_ONE_PT, &DlgLayoutCubeDemo::OnBnClickedOnePt)
	ON_BN_CLICKED(IDC_RADIO_TWO_PT, &DlgLayoutCubeDemo::OnBnClickedTwoPts)
	ON_BN_CLICKED(IDC_RADIO_MULTI_PT, &DlgLayoutCubeDemo::OnBnClickedRadioMultiPt)
END_MESSAGE_MAP()

void DlgLayoutCubeDemo::OnBnClickedOnePt()
{
	m_nInputWay = 0;
}

void DlgLayoutCubeDemo::OnBnClickedTwoPts()
{
	m_nInputWay = 1;
}


BOOL DlgLayoutCubeDemo::OnInitDialog()
{
	((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(TRUE);
	return TRUE;
}


// DlgLayoutCubeDemo 消息处理程序


void DlgLayoutCubeDemo::OnBnClickedRadioMultiPt()
{
	m_nInputWay = 2;
}
