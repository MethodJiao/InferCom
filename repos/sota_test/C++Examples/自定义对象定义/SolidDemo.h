#pragma once

namespace DemoObject
{ 
	//定义智能指针、引用等
	class SolidDemo;
	typedef SolidDemo const&          SolidDemoCR;
	typedef SolidDemo&                SolidDemoR;
	typedef SolidDemo*                SolidDemoP;
	typedef RefCountedPtr<SolidDemo>  SolidDemoPtr;

	class SolidDemo
	: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)	

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_SOLID_Demo; };
		//往数据库中写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//从数据库中读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics) override;

		//设置捕捉点
		virtual BIMBase::BPSnapStatus  _onSnap(BIMBase::Core::BPSnapContext& snapContext);

		std::vector<GePoint3d> m_points;
		p3d::GeSolidBaseType   m_solidType;     

	public:
		SolidDemo();
		~SolidDemo();

		//获取点集
		std::vector<GePoint3d>    getPoints()const{ return m_points; }
		void                      setPoints(std::vector<GePoint3d>& points);
		//获取类型
		p3d::GeSolidBaseType     getSolidType()const{ return m_solidType; }
		void                     setSolidType(p3d::GeSolidBaseType solidType){ m_solidType = solidType; }
		Demo_CREATE(SolidDemo);
		

	private:
		bool createTorusPipe(BPGraphicsPtr ptrGraphic);
		bool createCone(BPGraphicsPtr ptrGraphic);
		bool createBox(BPGraphicsPtr ptrGraphic);
		bool createSphere(BPGraphicsPtr ptrGraphic);
		bool createExtrusion(BPGraphicsPtr ptrGraphic);
		bool createRotationSweep(BPGraphicsPtr ptrGraphic);
		bool createRuledSweep(BPGraphicsPtr ptrGraphic);
	};
	Demo_EXTENSION(SolidDemo);
}

