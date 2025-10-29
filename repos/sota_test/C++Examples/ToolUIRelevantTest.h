#pragma once
/** @class
*  @brief   UI相关的一些范例，包括dock停靠框、视图右键等
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2021/4/12
*  ------------------------------------------------------------
*  @note:  -
*/


#include "afxdialogex.h"
#include "resource.h"

// 创建对话框ExternPaneDlg
class ExternPaneDlg : public CDialogEx
{
public:
	ExternPaneDlg();
	virtual ~ExternPaneDlg();

	DECLARE_DYNAMIC(ExternPaneDlg)
	enum { IDD = IDD_DIALOG_Dock };
	BOOL ExternPaneDlg::OnInitDialog();
	DECLARE_MESSAGE_MAP()

};


//ExternPaneDlg嵌入dock中
class ToolUIRelevantTest
{
public:
	ToolUIRelevantTest();
	~ToolUIRelevantTest();

	static void initDockTest();
	static void rButtonClickTest();

};



