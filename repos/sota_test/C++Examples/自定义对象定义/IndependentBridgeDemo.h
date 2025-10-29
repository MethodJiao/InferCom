#pragma once

/**
@brief   独立桥架造型，包括多种数据类型创建范例
@file    IndependentBridge.h
@author  北京构力科技股份有限公司
@date    2021.07.27
@note    常规类型：int、double、bool、string、二进制、数组、struct嵌套
@copyright Copyright (C), 2019-2028, Beijing Glory PKPM Technology. Co., Ltd.
*/

namespace DemoObject
{
	/** @brief 桥架类型*/
	enum IBPattern
	{
		enArchBridge = 0,	/** @brief 弧形桥架*/
		enParallelBridge,   /** @brief 平行桥架*/
	};
	class Tube;
	class   IndependentBridge;
	typedef IndependentBridge const& IndependentBridgeCR;
	typedef IndependentBridge& IndependentBridgeR;
	typedef IndependentBridge* IndependentBridgeP;
	typedef RefCountedPtr<IndependentBridge>  IndependentBridgePtr;

	/**
	@brief   独立桥架造型
	*/
	class IndependentBridge : public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)
	public:
		IndependentBridge();
		~IndependentBridge();

		/**@brief 设置类型(TYPE) */
		void setName(CString wsName);
		/**@brief 获取类型(TYPE) */
		CString getName() const;


		void setIBPattern(IBPattern enPattern);
		IBPattern getIBPattern() const;

		void setTubes(std::vector<TubePtr> gTubes);
		std::vector<TubePtr> getTubes() const;

		void setColumnDiameter(double dColumnDiameter);
		double getColumnDiameter() const;

		void setColumnHight(double dColumnHight);
		double getColumnHight() const;

		void setBridgeArchHight(double dBridgeArchHight);
		double getBridgeArchHight() const;

		void setNumRows(int nNumRows);
		int getNumRows() const;

		void setNumColumns(int nNumColumns);
		int getNumColumns() const;

		void setCSSLong(double dCSSLong);
		double getCSSLong() const;

		void setCSSWidth(double dCSSWidth);
		double getCSSWidth() const;

		void setCSSHight(double dCSSHight);
		double getCSSHight() const;

		void setTopSlabThickness(double dTopSlabThickness);
		double getTopSlabThickness() const;

		void setSideSlabThickness(double dSideSlabThickness);
		double getSideSlabThickness() const;

		void setTubeDiameter(vector<double> dTubeDiameters);
		/**
		@brief   获取电缆管内径
		*/
		vector<double> getTubeDiameter()const;

		void setTubeThickness(vector<double> dTubeThickness);
		/**
		@brief   获取电缆管壁厚
		*/
		vector<double> getTubeThickness() const;

		void setTubeCenters(pvector<GePoint3d>);
		pvector<GePoint3d> getTubeCenters();

	protected:
		/**
		@brief   通知基类表名
		*/
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };

		/**
		@brief   通知基类类名
		*/
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_INDEPENDENT_BRIDGE; };

		/**
		@brief   从数据库读取数据
		*/
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;

		/**
		@brief   写数据到数据库
		*/
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;

		/**
		@brief   创建集合造型
		*/
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics) override;

	private:
		bool __createBridteArch(BPGraphicsPtr&, int, double, int);
		void __mirror(BPGraphicsPtr&, IGeSolidBasePtr, GeTransform, GeTransform);
		void __updateTubes();

	private:
		CString          m_wsName;
		IBPattern         m_enPattern;
		std::vector<TubePtr> m_gTubes;

		double            m_dColumnDiameter;
		double            m_dColumnHight;
		double            m_dBridgeArchHight;
		int               m_nNumRows;
		int				  m_nNumColumns;
		double            m_dCSSLong;
		double            m_dCSSWidth;
		double            m_dCSSHight;
		double            m_dTopSlabThickness;
		double            m_dSideSlabThickness;

		Demo_CREATE(IndependentBridge)
	};
	Demo_EXTENSION(IndependentBridge)
};

