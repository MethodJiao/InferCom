#pragma once
/** @class
 *  @brief   剖切处理
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
	class BPDrawingCuttingDemo;
	typedef BPDrawingCuttingDemo const& BPDrawingCuttingDemoCR;
	typedef BPDrawingCuttingDemo& BPDrawingCuttingDemoR;
	typedef BPDrawingCuttingDemo* BPDrawingCuttingDemoP;
	typedef RefCountedPtr<BPDrawingCuttingDemo> BPDrawingCuttingDemoPtr;

	class BPDrawingCuttingDemo
	{
	public:
		BPDrawingCuttingDemo();
		~BPDrawingCuttingDemo();
		static BPDrawingCuttingDemoR Get();
		void getAllModelRange(BPProjectP pProject,GeRange3d& range);
		void getPhysicalModelElements(BPProjectP pProject,BPModelP model, p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>&cutinstance);
		void cutting(PBBimCore::PBModelInfoPtr modelInfoPtr/*CString drawingModelName*/,p3d::pvector<pair<BPDataKey, BPGraphicsPtr>>cutinstance, GePlane3d clipPlane, GeTransform sectionBox, GeTransform tm, BPProjectP pProject);//做剖切
		//vecplane代表剖切面，0是xy平面平面剖，1是yz平面立面剖，2是xz平面
		void __createSectionBoxAndClipPlane(int vecPlane, GeRange3d range, GePlane3d& clipplane, GeTransform& sectionBox, GeTransform& transform);

		PBBimCore::PBModelInfoPtr  getModelInfo(PString sName);//创建图纸model

		void addElementToCut(BPProjectP pProject, BPModelP model,BPEntity entity, PBBimCore::PBModelInfoPtr drawmodelInfoPtr, BPDrawingParasManagerDemo::eDrawingview type);


	};
}
