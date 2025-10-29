#include "stdafx.h"
#include "DlgArchWallDemo.h"
#include "afxdialogex.h"


// DlgArchWallDemo 对话框

IMPLEMENT_DYNAMIC(DlgArchWallDemo, CDialogEx)

DlgArchWallDemo::DlgArchWallDemo(CWnd* pParent /*=NULL*/)
	: CDialogEx(DlgArchWallDemo::IDD, pParent)
{
	m_nInputWay = 0;
}

DlgArchWallDemo::~DlgArchWallDemo()
{
}

void DlgArchWallDemo::updateUI()
{
	if (m_nInputWay == 0)
	{
		((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(TRUE);
		((CButton*)GetDlgItem(IDC_RADIO_TWO_PT))->SetCheck(false);
		((CButton*)GetDlgItem(IDC_RADIO_MULTI_PT))->SetCheck(false);
	}
	else
	{
		((CButton*)GetDlgItem(IDC_RADIO_TWO_PT))->SetCheck(TRUE);
		((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(false);
		((CButton*)GetDlgItem(IDC_RADIO_MULTI_PT))->SetCheck(false);
	}
}

void DlgArchWallDemo::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(DlgArchWallDemo, CDialogEx)
	ON_BN_CLICKED(IDC_RADIO_ONE_PT, &DlgArchWallDemo::OnBnClickedOnePt)
	ON_BN_CLICKED(IDC_RADIO_TWO_PT, &DlgArchWallDemo::OnBnClickedTwoPts)
	ON_BN_CLICKED(IDC_RADIO_MULTI_PT, &DlgArchWallDemo::OnBnClickedRadioMultiPt)
END_MESSAGE_MAP()

void DlgArchWallDemo::OnBnClickedOnePt()
{
	m_nInputWay = 0;
}

void DlgArchWallDemo::OnBnClickedTwoPts()
{
	m_nInputWay = 1;
}


BOOL DlgArchWallDemo::OnInitDialog()
{
	((CButton*)GetDlgItem(IDC_RADIO_ONE_PT))->SetCheck(TRUE);
	return TRUE;
}


// DlgArchWallDemo 消息处理程序
void DlgArchWallDemo::OnBnClickedRadioMultiPt()
{
	m_nInputWay = 2;
}
