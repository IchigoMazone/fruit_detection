from pydantic import BaseModel, Field, ConfigDict, validate_call
from typing import Literal, Any, TypeAlias, Annotated
from datetime import datetime
from tqdm import tqdm
import time, pytz

TotalType = Annotated[int | None, Field(ge=0)]
DescType = Annotated[str | None, Field(min_length=0, max_length=30)]
MinintervalType = Annotated[float | None, Field(ge=0, le=60)]
WidthType = Annotated[int, Field(ge=30, le=30)]
AmountType = Annotated[int | float, Field(ge=0)]
TimezoneType = Annotated[str, Field(init=False)]
UnitType: TypeAlias = Literal["it", "B", "file", "img", "batch", "sample", "epoch", "step"]

class Progress(BaseModel):

    model_config: Any = ConfigDict(extra="allow")
    total: TotalType = None
    pbar: str | None = None
    start_time: str | None = None
    bar_format: str | None = None
    desc: DescType = "Processing"
    unit: UnitType = "B"
    unit_scale: bool = True
    timezone: TimezoneType = "Asia/Ho_Chi_Minh"
    ascii: str | bool | None = None
    mininterval: MinintervalType = 0
    dynamic_ncols: bool = False
    refresh: bool = False
    width: WidthType = 30
    location: Any = pytz.timezone(timezone)

    @validate_call
    def start(
        self, 
        total: int | None = None,   
        bar_format: str | None = None, 
        ascii: str | bool | None = None, 
        unit: str | None = None
    ):
        bar_format_default = f"{{l_bar}}{{bar:{self.width}}}{{r_bar}}"
        self.bar_format = None if bar_format is None and self.width is None else bar_format_default
        self.total = total
        self.start_time = time.time()
        self.pbar= tqdm(
            total=self.total,
            desc=self.desc,
            unit=unit if unit is not None else self.unit,
            unit_scale=self.unit_scale,
            bar_format=bar_format if bar_format else self.bar_format,
            ascii=ascii if ascii else self.ascii,
            mininterval=self.mininterval,
            dynamic_ncols=self.dynamic_ncols,
            **(self.__pydantic_extra__ or {})
        )
        return self

    @validate_call
    def update(self, amount: AmountType):
        if self.pbar: 
            self.pbar.update(amount)

    @validate_call
    def set_metrics(self, refresh: bool = False, **kwargs):
        if self.pbar:
            self.pbar.set_postfix(
                **kwargs, 
                refresh=refresh if refresh is not None else self.refresh
            )

    @validate_call
    def finish(self, desc: DescType = None, content: str | None = None, end: bool = False):
        if self.pbar:
            self.pbar.close()
            elapsed = time.time() - self.start_time
            if end:
                print(f"\n{self.desc if not desc else desc} {'done in' if not content else content} {elapsed:.2f}s")

    @property
    def now(self):
        return datetime.now(self.location).strftime("%d/%m/%Y-%H:%M:%S")



if __name__ == "__main__":

    progress = Progress(**{
        "colour": None, 
        "ascii": None, 
        "width": 30
    })

    tracker = progress.start(**{
        "total": 10, 
        "ascii": " ░▒▓█", 
        "bar_format": "{l_bar}{bar:30}"
    })

    for i in range(10):
        tracker.set_metrics(**{
            "step": tracker.pbar.n, 
            "loss": 0.5
        })
        tracker.update(**{
            "amount": 1
        })
        time.sleep(1)

    tracker.finish()

    tracker = progress.start(total=5)
    for i in range(5):
        tracker.update(1)
        time.sleep(1)
    tracker.finish()


