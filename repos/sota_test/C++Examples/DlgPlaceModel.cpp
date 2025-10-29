
#include "stdafx.h"
#include "DLExamples.h"
#include "DlgPlaceModel.h"
#include "afxdialogex.h"

// DlgPlaceModel 对话框

IMPLEMENT_DYNAMIC(DlgPlaceModel, CDialogEx)

static DlgPlaceModel* s_pDlg;

//注册启动对话框的命令，通过showPlaceModel启动对话框，通过点击“绘制按钮”启动模型布置工具
static void ShowPlaceModel()
{
	PBBimModuleResourceOverride resOverride;
	CView* pView = ((CFrameWnd*)(AfxGetApp()->m_pMainWnd))->GetActiveFrame()->GetActiveView();
	if (s_pDlg == nullptr)
	{
		s_pDlg = new DlgPlaceModel;
		s_pDlg->Create(DlgPlaceModel::IDD, pView);
		s_pDlg->ShowWindow(SW_SHOW);
	}
	else
		s_pDlg->ShowWindow(SW_SHOW);
}

AutoDoRegisterFunctionsBegin
BIMBase::BPToolsManager::registerFun("showPlaceModel", &ShowPlaceModel);
AutoDoRegisterFunctionsEnd


DlgPlaceModel::DlgPlaceModel(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG_PLACEMODEL, pParent)
{

}

DlgPlaceModel::~DlgPlaceModel()
{
}

void DlgPlaceModel::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(DlgPlaceModel, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON3, &DlgPlaceModel::OnBnClickedButton3)
	ON_WM_PAINT()
END_MESSAGE_MAP()

// DlgPlaceModel 消息处理程序
BOOL DlgPlaceModel::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  在此添加额外的初始化

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 异常: OCX 属性页应返回 FALSE
}

void DlgPlaceModel::OnClose()
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	__super::OnClose();
	BPUserInputManager::exeCommand("$P3D.RestartSelTool");
}

void DlgPlaceModel::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型15.bfa";
	std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
	std::string location;
	location = wstring2string(bfaPath);
	P3DGlobalVariableManager::GetManager().SetValue("PlaceModelSelectLocation", location.c_str());

	BPUserInputManager::exeCommand("TestPlaceModelTool");
	TOOLSETTING_UPDATEVALUE;

}

void DlgPlaceModel::_UpdateUI()
{
	TOOLSETTING_UPDATEVALUE;
}

void DlgPlaceModel::OnPaint()
{
	CPaintDC dc(this);
	std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型15.bmp";
	std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
	HRESULT hres = m_image.Load(bfaPath.c_str());
	CRect rect;
	CWnd* pWnd = GetDlgItem(IDC_STATIC_PICTURE);
	if (!pWnd)
	{
		return;
	}
	pWnd->GetWindowRect(&rect);
	ScreenToClient(rect);

	m_image.Draw(dc.m_hDC, rect);

	__super::OnPaint();
}
