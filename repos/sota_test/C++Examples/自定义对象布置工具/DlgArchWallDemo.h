#pragma once
#include "resource.h"
/** @class
 *  @brief   立方体布置对话框
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/5/13
 *  ------------------------------------------------------------
 *  @note:  -
 */


class DlgArchWallDemo : public CDialogEx
{
	DECLARE_DYNAMIC(DlgArchWallDemo)

public:
	DlgArchWallDemo(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~DlgArchWallDemo();

	// 对话框数据
	enum { IDD = IDD_DLG_LAYOUT_WALL };

public:
	void updateUI();

public:
	int m_nInputWay;

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()

	afx_msg void OnBnClickedOnePt();
	afx_msg void OnBnClickedTwoPts();
	afx_msg void OnBnClickedRadioMultiPt();
};