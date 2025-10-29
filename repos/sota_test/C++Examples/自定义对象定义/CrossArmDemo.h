#pragma once
/** @class
*  @brief   创建塔架的横担
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  -
*/

namespace DemoObject
{ 
	//定义智能指针、引用等
	class CrossArmDemo;
	typedef CrossArmDemo const&          CrossArmDemoCR;
	typedef CrossArmDemo&                CrossArmDemoR;
	typedef CrossArmDemo*                CrossArmDemoP;
	typedef RefCountedPtr<CrossArmDemo>  CrossArmDemoPtr;

	class CrossArmDemo
	: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		CrossArmDemo();
		~CrossArmDemo();

		//横担主角钢截面
		p3d::GeCurveArrayPtr    getMainSection() const;
		void                 setMainSection(p3d::GeCurveArrayPtr  sectionCurve);

		//横担副角钢截面
		p3d::GeCurveArrayPtr    getAppendSection() const;
		void                 setAppendSection(p3d::GeCurveArrayPtr  sectionCurve);

		//横担单边支出宽度
		int    getOutWidth() const;
		void   setOutWidth(int nWidth);

		//横担高度
		int    getHeight() const;
		void   setHeight(int nHeight);

		//横担顶厚度
		int    getTopThickness() const;
		void   setTopThickness(int nVal);

		//横担底厚度
		int    getBaseThickness() const;
		void   setBaseThickness(int nVal);

		//端点厚度
		int		getEdgeThickness() const;
		void   setEdgeThickness(int nVal);

		//方向（up:true, down:false）
		bool    getDirection() const;
		void   setDirection(bool bDirection);

		//单支节点数
		int    getNodeCount() const;
		void   setNodeCount(int nCount);

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_CROSSARM_Demo; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics) override;

	private:
		p3d::IGeSolidBasePtr createsolid(p3d::GePoint3d sPoint, p3d::GePoint3d ePoint, p3d::GeCurveArrayPtr sectionCurve, bool bAdjustPoint = false);
		p3d::GeCurveArrayPtr initSectionBase(int nWidth);

	private:
		p3d::GeCurveArrayPtr  m_ptrMainSectionCurve;
		p3d::GeCurveArrayPtr  m_ptrAppendSectionCurve;
		int    m_nWidth;
		bool   m_bDirection;
		int    m_nHeight;
		int    m_nTopThickness;
		int    m_nBaseThickness;
		int    m_nEdgeThickness;
		int    m_nNode;

		Demo_CREATE(CrossArmDemo);
	};
	Demo_EXTENSION(CrossArmDemo);
}

