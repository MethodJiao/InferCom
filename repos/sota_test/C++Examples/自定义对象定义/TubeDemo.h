#pragma once

/**
@brief   圆管
@file    Tube.h
@author  北京构力科技股份有限公司
@date    2021.07.27
@copyright Copyright (C), 2019-2028, Beijing Glory PKPM Technology. Co., Ltd.
*/

namespace DemoObject
{
	class  Tube;
	typedef Tube const& TubeCR;
	typedef Tube& TubeR;
	typedef Tube* TubeP;
	typedef RefCountedPtr<Tube>  TubePtr;

	class Tube : public /*PBBimCore::PBNonGraphicObject*/BIMBase::Data::BPNonGraphicElement
	{
		DefineSuper(BPNonGraphicElement)
	public:
		Tube(p3d::GePoint3d ptCenter, double dTubeDiameter, double dTubeThickness);
		Tube();

		bool setCenter(p3d::GePoint3d ptCenter);
		p3d::GePoint3d getCenter() const;

		bool setDiameter(double dTubeDiameter);
		double getDiameter() const;

		bool setThickness(double dTubeThickness);
		double getThickness() const;

		TubePtr deepClone();

	protected:
		/**
		@brief   通知基类表名
		*/
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };

		/**
		@brief   通知基类类名
		*/
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_TUBE_Demo; };

		/**
		@brief   从数据库读取数据
		*/
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;

		/**
		@brief   写数据到数据库
		*/
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;

	private:
		GePoint3d m_ptCenter;
		double m_dTubeDiameter;
		double m_dTubeThickness;

		Demo_CREATE(Tube)
	};
	Demo_EXTENSION(Tube);
}
