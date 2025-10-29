#pragma once
/** @class
*  @brief   创建表格
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/4/26
*  ------------------------------------------------------------
*  @note:  -
*/


class ToolCreateTableDemo
{
public:
	ToolCreateTableDemo();
	~ToolCreateTableDemo();
	
	void createTable();
	void getCube(vector<DemoObject::CubeDemo> &vctCube);
	vector<DemoObject::UniversalBeamDemo> getUB();
};
