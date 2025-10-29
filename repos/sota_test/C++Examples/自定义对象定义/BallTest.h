#pragma once
/** @class
*  @brief   球造型
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note: 存的是世界坐标系下的点，一般情况下造型存局部坐标系下的点，具体可参考cubeTest
*/

namespace TestObject
{ 
	//定义智能指针、引用等
	class BallTest;
	typedef BallTest const&          BallTestCR;
	typedef BallTest&                BallTestR;
	typedef BallTest*                BallTestP;
	typedef RefCountedPtr<BallTest>  BallTestPtr;

	class BallTest
	: public BIMBase::Data::BPGraphicElement
	{
		DefineSuper(BPGraphicElement)

	public:
		BallTest();
		~BallTest();

		void setOrigin(GePoint3d point){ m_pOrigin = point; }
		GePoint3d getOrigin(){ return m_pOrigin; }

	protected:
		virtual Utf8String	 _getSchemaName() const override { return PBM_SCHEMA_TEST; };
		virtual Utf8String	 _getClassName() const override { return PBM_CLASS_BALL_TEST; };
		virtual ::p3d::P3DStatus      _copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const override;
		virtual ::p3d::P3DStatus      _initFromData(::BIMBase::Core::BPDataCR data) override;
		virtual BIMBase::Core::BPGraphicsPtr    _createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool bIsDynamics) override;
	private:
		GePoint3d m_pOrigin;

		TEST_CREATE(BallTest);
	};
	TEST_EXTENSION(BallTest);
}

