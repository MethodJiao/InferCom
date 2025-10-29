#pragma once

#include "PBBimTools\IToolProperty.h"
class InsulatorPropertyDemo :
	public IToolProperty
{
	enum InsulatorPropName
	{
		N,
		/**单串绝缘子片数量*/
		N1,
		/**绝缘子单片连接高度*/
		H1,
		/**大伞裙半径*/
		R1,
		/**小伞裙半径*/
		R2,
		/**绝缘子串半径*/
		R,
		/**双串间距*/
		D,
		/**前端长度（构架端*/
		FL,
		/**后端长度（导线端）*/
		AL,

		InsulatorPropCount
	};

public:
	InsulatorPropertyDemo();
	~InsulatorPropertyDemo();

	//获取属性并且在属性框显示
	virtual void OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)  override;
	//设置属性框中的值
	virtual TIErrorStatus OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item) override;
};

