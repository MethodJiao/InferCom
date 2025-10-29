#pragma once
/** @class
*  @brief   圆管造型
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
	struct  ExampleTubeDemo;
	typedef ExampleTubeDemo const& ExampleTubeDemoCR;
	typedef ExampleTubeDemo& ExampleTubeDemoR;
	typedef ExampleTubeDemo* ExampleTubeDemoP;
	typedef RefCountedPtr<ExampleTubeDemo>  ExampleTubeDemoPtr;

	struct ExampleTubeDemo : public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)
	public:
		ExampleTubeDemo(GePoint3d ptStart,GePoint3d ptEnd, double dTubeDiameter, double dTubeThickness);
		ExampleTubeDemo();

		void setStartPoint(GePoint3d ptStart);
		GePoint3d getStartPoint() const;

		void setEndPoint(GePoint3d ptEnd);
		GePoint3d getEndPoint() const;

		void setDiameter(double dTubeDiameter);
		double getDiameter() const;

		void setThickness(double dTubeThickness);
		double getThickness() const;

	protected:
		/**
		@brief   通知基类表名
		*/
		virtual  Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };

		/**
		@brief   通知基类类名
		*/
		virtual Utf8String	 _getClassName() const override { return "ExampleTubeDemo"; };

		/**
		@brief   从数据库读取数据
		*/
		virtual ::p3d::P3DStatus      _initFromData(::BIMBase::Core::BPDataCR data) override;

		/**
		@brief   写数据到数据库
		*/
		virtual ::p3d::P3DStatus      _copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const;

		/**
		@brief   创建集合造型
		*/
		virtual BIMBase::Core::BPGraphicsPtr    _createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool bIsDynamics) override;

	private:
		GePoint3d m_ptStart;
		GePoint3d m_ptEnd;
		double m_dTubeDiameter;
		double m_dTubeThickness;

		Demo_CREATE(ExampleTubeDemo)
	};
	Demo_EXTENSION(ExampleTubeDemo)
}
