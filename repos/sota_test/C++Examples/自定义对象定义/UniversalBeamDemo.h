#pragma once
/** @class
*  @brief   工字钢造型
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note:  存的是局部坐标系下的信息
*/

namespace DemoObject
{
	//定义智能指针、引用等
	class UniversalBeamDemo;
	typedef UniversalBeamDemo const& UniversalBeamDemoCR;
	typedef UniversalBeamDemo& UniversalBeamDemoR;
	typedef UniversalBeamDemo* UniversalBeamDemoP;
	typedef RefCountedPtr<UniversalBeamDemo>  UniversalBeamDemoPtr;

	class UniversalBeamDemo
		: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		UniversalBeamDemo();
		~UniversalBeamDemo();

		int getWidth() const;
		void   setWidth(int nWidth);

		int getLength() const;
		void   setLength(int nLength);

		int getThick() const;
		void   setThick(int nThick);
		
		int getInnerHeight() const;
		void   setInnerHeight(int nInnerHeight);

		//获取起点
		GePoint3d    getStartPoint() const;
		//获取上部起点
		GePoint3d    getUpStartPoint() const;
		//获取中点
		GePoint3d    getCenterPoint() const;
		//获取终点
		GePoint3d    getEndPoint() const;
		//获取上部终点
		GePoint3d    getUpEndPoint() const;

		BIMBase::Core::BPGraphicsPtr createGraphicsPlane(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId);
	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_UNIVERSAL_BEAM_Demo; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;

		//设置捕捉点
		virtual BIMBase::BPSnapStatus  _onSnap(BIMBase::Core::BPSnapContext& snapContext);

	private:
		int m_nThick;
		int m_nWidth;
		int m_nLenght;
		int m_nInnerHeight;


		Demo_CREATE(UniversalBeamDemo);
	};
	Demo_EXTENSION(UniversalBeamDemo);
}

