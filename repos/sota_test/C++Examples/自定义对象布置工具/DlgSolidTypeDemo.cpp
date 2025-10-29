// DlgSolidTypeDemo.cpp : 实现文件
//

#include "stdafx.h"
#include "Examples.h"
#include "DlgSolidTypeDemo.h"
#include "afxdialogex.h"


// DlgSolidTypeDemo 对话框

IMPLEMENT_DYNAMIC(DlgSolidTypeDemo, CDialogEx)

DlgSolidTypeDemo::DlgSolidTypeDemo(CWnd* pParent /*=NULL*/)
	: CDialogEx(DlgSolidTypeDemo::IDD, pParent)
{
	m_eType = GeSolidBaseType::GeSolidBaseType_TorusPipe;
}

DlgSolidTypeDemo::~DlgSolidTypeDemo()
{
}

void DlgSolidTypeDemo::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, m_cmbType);
}


BEGIN_MESSAGE_MAP(DlgSolidTypeDemo, CDialogEx)
	ON_CBN_SELCHANGE(IDC_COMBO1, &DlgSolidTypeDemo::OnCbnSelchangeCombo1)
END_MESSAGE_MAP()



void DlgSolidTypeDemo::OnCbnSelchangeCombo1()
{
	m_eType = (GeSolidBaseType)(m_cmbType.GetCurSel() + 1);
	switch (m_eType)
	{
	case p3d::GeSolidBaseType::GeSolidBaseType_None:
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_TorusPipe:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择大圆圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Cone:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择底面圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Box:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择盒体位置"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Sphere:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择球心"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_Extrusion:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"将布置一个圆环，请选择圆环底面圆心"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RotationalSweep:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"将布置一个壳体，请选择圆心点"));
		break;
	case p3d::GeSolidBaseType::GeSolidBaseType_RuledSweep:
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请在高度方向选择扫掠第一点"));
		break;
	default:
		break;
	}
}


BOOL DlgSolidTypeDemo::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	m_cmbType.InsertString(0, _T("圆管体"));
	m_cmbType.InsertString(1, _T("锥台体"));
	m_cmbType.InsertString(2, _T("盒体"));
	m_cmbType.InsertString(3, _T("球体"));
	m_cmbType.InsertString(4, _T("拉伸体"));
	m_cmbType.InsertString(5, _T("旋转体"));
	m_cmbType.InsertString(6, _T("扫掠体"));
	m_cmbType.SetCurSel(0);

	return TRUE;  // return TRUE unless you set the focus to a control
	// 异常:  OCX 属性页应返回 FALSE
}
