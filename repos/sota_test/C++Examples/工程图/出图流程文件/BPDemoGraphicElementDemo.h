#pragma once
/** @class
*  @brief   剖切对象临时用的基类
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  
*/

namespace DemoObject
{
	//定义智能指针、引用等
	class BPDemoGraphicElementDemo;
	typedef BPDemoGraphicElementDemo const& BPDemoGraphicElementDemoCR;
	typedef BPDemoGraphicElementDemo& BPDemoGraphicElementDemoR;
	typedef BPDemoGraphicElementDemo* BPDemoGraphicElementDemoP;
	typedef RefCountedPtr<BPDemoGraphicElementDemo>  BPDemoGraphicElementDemoPtr;

	class BPDemoGraphicElementDemo
		: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		BPDemoGraphicElementDemo();
		~BPDemoGraphicElementDemo();
		BIMBase::Core::BPGraphicsPtr createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId);

	protected:
		
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId);
		virtual ::p3d::P3DStatus                _addToProject(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId) override;
		virtual ::p3d::P3DStatus                _replaceInProject(::BIMBase::Core::BPProjectR project, bool bReCreateGeometry = true) override;
	
	};
}


