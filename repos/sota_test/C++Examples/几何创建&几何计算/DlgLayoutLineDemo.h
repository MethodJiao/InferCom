#pragma once
/** @class
*  @brief   绘制线条对话框
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  -
*/
// DlgLayoutLineDemo 对话框

class DlgLayoutLineDemo : public CDialogEx
{
	DECLARE_DYNAMIC(DlgLayoutLineDemo)

public:
	DlgLayoutLineDemo(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~DlgLayoutLineDemo();

// 对话框数据
	enum { IDD = IDD_DLG_LAYOUT_LINE };

public:
	int m_nInputWay;

public:
	afx_msg void OnBnClickedLayout();
	afx_msg void OnBnClickedDraw();

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持
	virtual BOOL OnInitDialog();

	DECLARE_MESSAGE_MAP()


private:
	CButton m_btLayout;
	CButton m_btDraw;
public:
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	afx_msg void OnBnClickedRadioContinuedraw();
	CButton m_btContinueDraw;
};
