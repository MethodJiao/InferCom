#pragma once
/** @class
*  @brief   创建塔架的立杆
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
	class TowerDemo;
	typedef TowerDemo const&          TowerDemoCR;
	typedef TowerDemo&                TowerDemoR;
	typedef TowerDemo*                TowerDemoP;
	typedef RefCountedPtr<TowerDemo>  TowerDemoPtr;

	class TowerDemo
	: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		TowerDemo();
		~TowerDemo();

		//获取第i个节点相关参数，用于布置横担
		bool getNodeParameter(IN int nIndex, OUT GePoint3dR nodePoint, OUT double& width);

		//立柱角钢截面
		p3d::GeCurveArrayPtr    getMainSection() const;
		void                 setMainSection(p3d::GeCurveArrayPtr  ptrSectionCurve);

		//横杆角钢截面
		p3d::GeCurveArrayPtr    getAppendSection() const;
		void                 setAppendSection(p3d::GeCurveArrayPtr  ptrSectionCurve);

		//底座边长
		int    getBaseWidth() const;
		void   setBaseWidth(int nWidth);

		//倾斜角度
		double getSlope() const;
		void   setSlope(double dSlope);

		//塔高
		int    getHeight() const;
		void   setHeight(int nHeight);

		//节点数
		int    getNodeCount() const;
		void   setNodeCount(int nCount);

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_TOWER_Demo; };
		//往数据库中写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//从数据库中读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics) override;

	private:
		p3d::IGeSolidBasePtr createsolid(p3d::GePoint3d sPoint, p3d::GePoint3d ePoint, p3d::GeCurveArrayPtr ptrSectionCurve, bool bAdjustPoint = false);
		p3d::GeCurveArrayPtr initSectionBase(int nWidth);
		void initPoints();

	private:
		p3d::GeCurveArrayPtr  m_ptrMainSectionCurve;
		p3d::GeCurveArrayPtr  m_ptrAppendSectionCurve;
		int    m_nWidth;
		double m_dSlope;
		int    m_nHeight;
		int    m_nNode;
		std::vector<std::vector<GePoint3d>> m_points;
		p3d::pvector<CrossArmDemoPtr>       m_vctCrossArm;
		p3d::pvector<pair<int, int>>       m_vctArmPara;

		Demo_CREATE(TowerDemo);
	};
	Demo_EXTENSION(TowerDemo);
}

