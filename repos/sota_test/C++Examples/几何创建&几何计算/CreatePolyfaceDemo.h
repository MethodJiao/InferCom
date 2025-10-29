#pragma once
/** @class
*  @brief   创建ployface，并且做布尔运算
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/4/26
*  ------------------------------------------------------------
*  @note:  -
*/


class CreatePolyfaceDemo
{
public:
	CreatePolyfaceDemo();
	~CreatePolyfaceDemo();

	static BPGraphicsPtr createHexahedron(double length, double width, double height);

	static void funCreateHexahedron();

	//创建地形面片
	static void funPolyface();
	//创建polyface实体
	static BPGraphicsPtr createPolyfaceSolid();
	static void funCreatePolyfaceSolid();

	//polyface实体与圆柱布尔运算
	static void doBoolean();
	
	//IGeSolidBase转PolyfaceHandle
	static bool solidBaseToPolyface(pvector<PolyfaceHandlePtr> meshData, IGeSolidBasePtr solidPrimitive);

	//两个polyface拼接成一个polyface
	static void combinePolyface();
	//static void createHexahedron();
};

