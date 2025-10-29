#pragma once
#include "afxwin.h"
/** @class
*  @brief  绘制各种形体对话框 
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  -
*/

// DlgSolidTypeDemo 对话框

class DlgSolidTypeDemo : public CDialogEx
{
	DECLARE_DYNAMIC(DlgSolidTypeDemo)

public:
	DlgSolidTypeDemo(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~DlgSolidTypeDemo();

// 对话框数据
	enum { IDD = IDD_DLGSOLIDTYPE };

public:
	GeSolidBaseType m_eType;
	CComboBox m_cmbType;

public:
	afx_msg void OnCbnSelchangeCombo1();
	virtual BOOL OnInitDialog();

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()

};
