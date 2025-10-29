#pragma once
/** @class
*  @brief   dock停靠框范例，包括通过命令显示停靠对话框，右键菜单显隐停靠对话框所需接口，其中添加右键菜单参见RButtonClickEvent
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




