import { ReactNode } from 'react';
import './Unit.css';

function Unit({ children, className }: { children: ReactNode, className?: string}) {
    return <>
        <div className="unit">
            <div className="bottom">
                <div className={"top " + className}>
                    {children}
                </div>
            </div>
        </div>
    </>;
}

export default Unit