#pragma once
#include<map>
#include<string>
#include <utility>
using namespace std;

/** @class
 *  @brief   出图参数管理
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */


	namespace DemoObject
{

	////定义智能指针、引用等
	class BPDrawingParasManagerDemo;
	typedef BPDrawingParasManagerDemo const& BPDrawingParasManagerDemoCR;
	typedef BPDrawingParasManagerDemo& BPDrawingParasManagerDemoR;
	typedef BPDrawingParasManagerDemo* BPDrawingParasManagerDemoP;

	struct Params{ 
		CString strFrame;
		CString strDrawingName;
		CString strLegend;
		CString strLabel;
		CString cutModelName;
		Params() {};
		
	};

	class BPDrawingParasManagerDemo
	{
	public:
		BPDrawingParasManagerDemo();
		~BPDrawingParasManagerDemo();

		enum eDrawingview
		{
			X_Y,   
			Y_Z, 
			X_Z
		};
		enum eLayoutType
		{
		};
		static BPDrawingParasManagerDemoR Get();
		void setParams(Params para) {
			m_param = para;
		}
		Params getParams() {
			return m_param;
		}
		void setDrawingview(eDrawingview type) {
			m_drawingView = type;
		}
		eDrawingview getDrawingview() {
			return m_drawingView;
		}
	private:
		std::map<string, std::pair<int, int>> m_pars;
		bool m_bNeedIntergreted;
		Params m_param;
		eDrawingview m_drawingView;
	};
}
