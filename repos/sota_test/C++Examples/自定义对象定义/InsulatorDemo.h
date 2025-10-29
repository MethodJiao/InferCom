#pragma once
/** @class
*  @brief   绝缘子串造型
*  @author  北京构力科技有限公司
*  @date    2023/3/31
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
	class InsulatorDemo;
	typedef InsulatorDemo const& InsulatorDemoCR;
	typedef InsulatorDemo& InsulatorDemoR;
	typedef InsulatorDemo* InsulatorDemoP;
	typedef RefCountedPtr<InsulatorDemo>  InsulatorDemoPtr;

	class InsulatorDemo
		: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		InsulatorDemo();
		~InsulatorDemo();

	
		int getN() const;
		void setN(const int N);
		double getD() const;
		void setD(const double D);
		int getN1() const;
		void setN1(const int N1);
		double getH1() const;
		void setH1(const double H1);
		double getR1()const;
		void setR1(const double R1);
		double getR2()const;
		void setR2(const double R2);
		double getR()const;
		void setR(const double R);
		double getFL() const;
		void setFL(const double FL);
		double getAL()const;
		void setAL(const double AL);
		/*int getLN() const;
		void setLN(const int LN);*/

		void setCenter(const GePoint3d ptCenter);
		GePoint3d    getCenter() const { return m_ptCenter; };

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_Demo; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_INSULATOR_Demo; };
		//写数据
		virtual ::p3d::P3DStatus _copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const override;
		//读数据
		virtual ::p3d::P3DStatus _initFromData(BIMBase::Core::BPDataCR	instance) override;
		//创建几何造型
		virtual BIMBase::Core::BPGraphicsPtr _createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics) override;



	private:
		/**联数*/
		int m_nN;
		/**单串绝缘子片数量*/
		int m_nN1;
		/**绝缘子单片连接高度*/
		double m_dH1;
		/**大伞裙半径*/
		double m_dR1;
		/**小伞裙半径*/
		double m_dR2;
		/**绝缘子串半径*/
		double m_dR;
		/**双串间距*/
		double m_dD;
		/**前端长度（构架端*/
		double m_dFL;
		/**后端长度（导线端）*/
		double m_dAL;
		long long m_temp;


		int m_nRed;
		int m_nGreen;
		int m_nBlue;
		double m_dAlpha;
		GePoint3d m_ptCenter;
		GeVec3d m_vtAxisX;
		GeVec3d m_vtAxisY;
		GeVec3d m_vtAxisZ;
		Demo_CREATE(InsulatorDemo);
	};
	Demo_EXTENSION(InsulatorDemo);
}

