#pragma once
/** @class
 *  @brief   出图信息处理
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
	class BPDrawingInfoDemo;
	typedef BPDrawingInfoDemo const& BPDrawingInfoDemoCR;
	typedef BPDrawingInfoDemo& BPDrawingInfoDemoR;
	typedef BPDrawingInfoDemo* BPDrawingInfoDemoP;
	typedef RefCountedPtr<BPDrawingInfoDemo> BPDrawingInfoDemoPtr;

	class BPDrawingInfoDemo
	{
	public:
		BPDrawingInfoDemo();
		~BPDrawingInfoDemo();
		static BPDrawingInfoDemoR Get();
		void drawFrame(PBModelInfoPtr modelInfo);
		void importFrame(PBModelInfoPtr modelInfo);
		void drawDimension(PBModelInfoPtr modelInfo);//画标注
		void drawTable(PBModelInfoPtr modelInfo);//画表格，表格里统计
		void layoutPic(std::map<CString, PBModelInfoPtr> modelinfo);
		void drawBlock(PBModelInfoPtr modelInfo);

	private:
		GePoint3d m_ptFrameInnerRU;
		GePoint3d m_ptFrameInnerRD;
		double m_dFrameScale;

	};
}
